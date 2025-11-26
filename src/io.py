import numpy as np
import os

def save_ply(path, points, colors, conf=None, conf_thresh=0.5):
    """
    Saves points as a binary PLY file.
    Filters points where conf <= conf_thresh if conf is provided.
    
    Args:
        path (str): Output filename
        points (np.array): Nx3 float array
        colors (np.array): Nx3 uint8 array (0-255)
        conf (np.array): N float array (optional)
        conf_thresh (float): Threshold for filtering
    """
    
    # 1. Filter if confidence provided
    if conf is not None:
        mask = conf > conf_thresh
        points = points[mask]
        colors = colors[mask]
    
    num_points = points.shape[0]
    if num_points == 0:
        print(f"Warning: No points to save to {path}")
        return

    # 2. Binary PLY Header
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    # 3. Write Binary Data
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        
        # Create structured array for interleaved writing (fastest method)
        # Format: float32, float32, float32, uint8, uint8, uint8
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        cloud_data = np.empty(num_points, dtype=dtype)
        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]
        cloud_data['red'] = colors[:, 0]
        cloud_data['green'] = colors[:, 1]
        cloud_data['blue'] = colors[:, 2]
        
        cloud_data.tofile(f)
