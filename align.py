'''
python align.py \
  --dataset_name t01_v01_s04_r08_ImageDensity_A01_iPad_LTS \
  --pi3_images_dir data/pi3/t01_v01_s04_r08_ImageDensity_A01_iPad_LTS/images \
  --conf_thresh 1.0
'''
import argparse
import os
import torch
import numpy as np
from src.loaders import load_colmap_data, load_pi3_results
from src.geometry import compute_umeyama, apply_sim3, compute_alignment_error
from src.io import save_ply

def main():
    parser = argparse.ArgumentParser(description="Align Pi3 Point Cloud to COLMAP Frame")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the dataset folder (e.g. t01_v01...)")
    parser.add_argument("--pi3_images_dir", type=str, required=True,
                        help="Path to the subset of images Pi3 processed (for filename matching)")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold for filtering Pi3 points")
    args = parser.parse_args()

    # --- 1. Setup Paths ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    colmap_dir = os.path.join(base_dir, "data", "colmap", args.dataset_name, "colmap_files")
    pi3_file = os.path.join(base_dir, "data", "pi3", args.dataset_name, "pi3_files", "results_pie.pt")
    output_dir = os.path.join(base_dir, "output", args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Aligning Dataset: {args.dataset_name} ---")

    # --- 2. Load COLMAP (Reference) ---
    colmap_cameras, pts_col, cols_col = load_colmap_data(colmap_dir)
    
    # Save sparse colmap.ply for reference
    print("Exporting colmap.ply (Reference)...")
    if pts_col is not None:
        save_ply(os.path.join(output_dir, "colmap.ply"), pts_col, cols_col)
    else:
        print("Warning: No sparse points found in COLMAP reconstruction.")
        # Create empty arrays to prevent crashes later during merge
        pts_col = np.empty((0, 3), dtype=np.float32)
        cols_col = np.empty((0, 3), dtype=np.uint8)

    # --- 3. Load Pi3 (Source) ---
    print(f"Loading Pi3 results from {pi3_file}...")
    pi3_points, pi3_conf, pi3_poses, pi3_colors = load_pi3_results(pi3_file)
    
    # Save unaligned pi3.ply for debugging
    print("Exporting pi3.ply (Unaligned)...")
    save_ply(os.path.join(output_dir, "pi3.ply"), 
             pi3_points.reshape(-1, 3).numpy(), 
             pi3_colors.reshape(-1, 3), 
             pi3_conf.reshape(-1).numpy(), 
             args.conf_thresh)

    # --- 4. Match Cameras (Subset Logic) ---
    pi3_image_names = sorted([f for f in os.listdir(args.pi3_images_dir) 
                              if f.lower().endswith(('.jpg', '.png'))])
    
    N_pi3 = pi3_points.shape[1]
    if len(pi3_image_names) != N_pi3:
        print(f"Error: Pi3 tensor has {N_pi3} frames, but found {len(pi3_image_names)} images.")
        return

    src_centers = [] # Pi3
    dst_centers = [] # COLMAP
    
    print(f"Matching {len(pi3_image_names)} Pi3 images against {len(colmap_cameras)} COLMAP images...")
    
    count = 0
    for idx, name in enumerate(pi3_image_names):
        if name in colmap_cameras:
            p_center = pi3_poses[0, idx, :3, 3].numpy()
            c_center = colmap_cameras[name]
            src_centers.append(p_center)
            dst_centers.append(c_center)
            count += 1

    if count < 3:
        raise ValueError("Error: Need at least 3 matching cameras to calculate alignment.")

    src_arr = np.array(src_centers)
    dst_arr = np.array(dst_centers)

    # --- 5. Compute Alignment ---
    print(f"Computing Sim(3) alignment using {count} shared cameras...")
    scale, R, t = compute_umeyama(src_arr, dst_arr)
    rmse, rel_error = compute_alignment_error(src_arr, dst_arr, scale, R, t)

    print(f"Alignment Found: Scale={scale:.4f}, RMSE={rmse:.4f} ({rel_error:.2f}%)")

    # --- 6. Apply Transform & Merge ---
    print("Applying transform to dense point cloud...")
    
    # A. Align Pi3 Points
    flat_pi3_pts = pi3_points.reshape(-1, 3).numpy()
    aligned_pi3_pts = apply_sim3(flat_pi3_pts, scale, R, t)
    
    flat_pi3_cols = pi3_colors.reshape(-1, 3)
    flat_pi3_conf = pi3_conf.reshape(-1).numpy()

    # B. Merge with COLMAP Points
    # To merge, we need to handle the confidence.
    # COLMAP points are "ground truth", so we give them max confidence (e.g. 100.0)
    # so they are never filtered out by conf_thresh.
    colmap_conf = np.ones(pts_col.shape[0], dtype=np.float32) * 100.0
    
    print(f"Merging {pts_col.shape[0]} COLMAP points with {aligned_pi3_pts.shape[0]} Pi3 points...")
    
    merged_pts = np.vstack((pts_col, aligned_pi3_pts))
    merged_cols = np.vstack((cols_col, flat_pi3_cols))
    merged_conf = np.concatenate((colmap_conf, flat_pi3_conf))
    
    # --- 7. Save Merged Result ---
    save_path = os.path.join(output_dir, "colmap_with_pi3.ply")
    print(f"Saving merged result to {save_path}...")
    
    save_ply(save_path,
             merged_pts,
             merged_cols,
             merged_conf,
             args.conf_thresh)

    print("Done. Visualization ready.")

if __name__ == "__main__":
    main()
