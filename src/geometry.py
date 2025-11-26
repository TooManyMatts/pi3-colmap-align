import numpy as np

def compute_umeyama(source_points, target_points):
    """
    Computes the Sim(3) transformation (s, R, t) that aligns source to target.
    Equation: Target ~= s * R @ Source + t
    
    Args:
        source_points (np.array): Nx3 array (Pi3 Camera Centers)
        target_points (np.array): Nx3 array (COLMAP Camera Centers)
        
    Returns:
        scale (float), Rotation (3x3), Translation (3,)
    """
    assert source_points.shape == target_points.shape
    n = source_points.shape[0]

    # 1. Center the points
    mu_s = np.mean(source_points, axis=0)
    mu_t = np.mean(target_points, axis=0)
    
    src_centered = source_points - mu_s
    tgt_centered = target_points - mu_t

    # 2. Compute Variance of Source
    sigma2 = np.mean(np.sum(src_centered ** 2, axis=1))

    # 3. Covariance Matrix
    H = (tgt_centered.T @ src_centered) / n
    
    # 4. SVD for Rotation
    U, D, Vt = np.linalg.svd(H)
    Rot = U @ Vt
    
    # Handle reflection case (det(R) should be 1, not -1)
    if np.linalg.det(Rot) < 0:
        Vt[-1, :] *= -1
        Rot = U @ Vt

    # 5. Compute Scale and Translation
    # Scale correction matrix for reflection case
    S_mat = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S_mat[2, 2] = -1

    scale = np.trace(np.diag(D) @ S_mat) / sigma2
    trans = mu_t - scale * (Rot @ mu_s)

    return scale, Rot, trans

def apply_sim3(points, scale, R, t):
    """
    Applies the Sim(3) transformation to a set of 3D points.
    Formula: p' = s * (p @ R^T) + t
    """
    # points is Nx3. R is 3x3.
    # We use R.T because we are multiplying points from the right
    return (scale * points @ R.T) + t

def compute_alignment_error(source_points, target_points, scale, R, t):
    """
    Calculates RMSE and Relative Error of the alignment.
    
    Returns:
        rmse (float): Root Mean Square Error in target units
        rel_error (float): RMSE normalized by target bounding box diagonal (%)
    """
    # 1. Apply transform to source
    src_aligned = apply_sim3(source_points, scale, R, t)
    
    # 2. Compute residuals
    diff = target_points - src_aligned
    squared_errors = np.sum(diff ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_errors))
    
    # 3. Compute Scene Scale (Bounding Box Diagonal)
    bbox_min = np.min(target_points, axis=0)
    bbox_max = np.max(target_points, axis=0)
    scene_diagonal = np.linalg.norm(bbox_max - bbox_min)
    
    # 4. Relative Error
    if scene_diagonal > 1e-6:
        rel_error = (rmse / scene_diagonal) * 100.0
    else:
        rel_error = 0.0
        
    return rmse, rel_error
