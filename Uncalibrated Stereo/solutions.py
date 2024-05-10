import numpy as np
import utils
from typing import Tuple

ransac_n = 500
ransac_eps = 1
window_size = 69


def estimate_P(uv: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """
    Estimate the projection matrix P from the given point correspondences.

    Args:
        uv (np.ndarray): image coordinates in pixels (Nx2)
        X_world (np.ndarray): world coordinates in meters (Nx3)

    Returns:
        np.ndarray: camera projection matrix (3x4)
    """

    assert uv.shape[0] == X_world.shape[0]
    assert uv.shape[1] == 2 and X_world.shape[1] == 3

    n = uv.shape[0]
    A = np.zeros((n * 2, 12))

    for i in range(n):
        x, y, z = X_world[i]
        u, v = uv[i]
        A[i * 2] = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
        A[i * 2 + 1] = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]

    _, _, V_mat = np.linalg.svd(A)
    p = V_mat[-1]
    P = p.reshape(3, 4)
    return P


def reprojection_error(uv: np.ndarray, 
                       X_world: np.ndarray, 
                       P: np.ndarray) -> float:
    """
    Compute the root-mean-squared (RMS) reprojection error over all the points.

    Args:
        uv (np.ndarray): image points (Nx2)
        X_world (np.ndarray): world points (Nx3)
        P (np.ndarray): camera projection matrix (3x4)

    Returns:
        float: RMS reprojection error
    """
    n = X_world.shape[0]
    ssd = 0

    for i in range(n):
        world = np.append(X_world[i], 1)
        proj = np.matmul(P, world)
        u_r = [proj[0] / proj[2], proj[1] / proj[2]]

        sd = (uv[i] - u_r)**2
        ssd += sd

    error = (ssd / n)**0.5
    return error[0]


def estimate_F(uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """
    Estimate the fundamental matrix mapping the points in uv1 and
    uv2. The fundamental matrix should map points in the first image (uv1) to
    lines in the second image (uv2).

    Since the fundamental matrix has arbitrary scale, it should
    be scaled such that ||f||_2 = 1.

    Args:
        uv1 (np.ndarray): image points in the first image (in pixels, Nx2)
        uv2 (np.ndarray): image points in the second image (in pixels, Nx2)

    Returns:
        np.ndarray: fundamental matrix (3x3)
    """

    assert uv1.shape[0] == uv2.shape[0]
    assert uv1.shape[1] == 2 and uv2.shape[1] == 2

    n = uv1.shape[0]
    A = np.zeros((n, 9))

    for i in range(n):
        u1, v1 = uv1[i]
        u2, v2 = uv2[i]
        A[i] = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1]

    _, _, V = np.linalg.svd(A)
    f = V[-1]
    F = f / np.linalg.norm(f)
    F = F.reshape(3, 3)
    return F


def point_to_epiline(pts1: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the epipolar line in the second image corresponding to each point
    in the first image.

    Args:
        pts1 (np.ndarray): points in the first image (Nx2)
        F (np.ndarray): fundamental matrix mapping points in the first image to
            lines in the second image (3x3)

    Returns:
        np.ndarray: lines in the second image (Nx3)
    """

    n = pts1.shape[0]
    lines = np.zeros((n, 3))
    for i in range(n):
        point = np.array([pts1[i][0], pts1[i][1], 1])
        lines[i] = np.dot(F, point)

    return lines


def error_F(uv1: np.ndarray, uv2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the mean distance from epiline to point on the second image.

    Args:
        uv1 (np.ndarray): image points in the first image (in pixels, Nx2)
        uv2 (np.ndarray): image points in the second image (in pixels, Nx2)
        F (np.ndarray): fundamental matrix mapping points in the first image to 
            lines in the second image (Nx3)

    Returns:
        np.ndarray: distances from epiline to point in the second image (Nx1).
    """
    lines = point_to_epiline(uv1, F)
    n = uv2.shape[0]

    distances = np.zeros(n)

    for i in range(n):
        u, v = uv2[i]
        line = lines[i]

        distances[i] = abs(line[0] * u + line[1] * v + line[2]) / np.sqrt(line[0]**2 + line[1]**2)

    return distances


def estimate_F_RANSAC(uv1: np.ndarray, 
                      uv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the fundamental matrix for the two camera views using RANSAC.
    Return the estimate of F, the list of inlier indices, and the best error
    computed using error_F().

    Args:
        uv1 (np.ndarray): image points in the first image (in pixels, Nx2)
        uv2 (np.ndarray): image points in the second image (in pixels, Nx2)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Fundamental matrix, inlier 
            indices, and distance error over the inliers.
    """

    num_points = uv1.shape[0]
    inliers = []
    F = None
    distances = None

    for _ in range(ransac_n):
        pt_indx = np.random.choice(num_points, 8, replace=False)
        src_set = uv1[pt_indx]
        dest_set = uv2[pt_indx]

        f = estimate_F(src_set, dest_set)

        curr_dist = error_F(uv1, uv2, f)
        curr_inliers = np.where(curr_dist < ransac_eps)[0]

        if len(curr_inliers) > len(inliers):
            inliers = curr_inliers
            F = f
            distances = curr_dist

    return F, inliers, distances


def estimate_E(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute the essential matrix from the fundamental matrix and intrinsics.

    Args:
        F (np.ndarray): fundamental matrix (3x3)
        K (np.ndarray): camera intrinsics (3x3)

    Returns:
        np.ndarray: essential matrix (3x3)
    """
    E = K.T @ F @ K
    return E


def similarity_metric(window_left: np.ndarray, 
                      window_right: np.ndarray) -> float:
    """
    Compute the similarity measure between two image windows.

    Args:
        window_left (np.ndarray): left image window
        window_right (np.ndarray): right image window

    Returns:
        float: value of the similarity measure
    """
    assert window_left.shape == window_right.shape
    ncc = np.sum(np.multiply(window_left, window_right)) / np.sqrt(np.sum(window_left**2) * np.sum(window_right**2))
    return ncc


def dense_point_correspondences(
    img_left: np.ndarray, 
    img_right: np.ndarray, 
    F: np.ndarray, 
    w: int,
    img_left_mask: np.ndarray) -> np.ndarray:
    """
    Find the coordinates of the point correspondence in img_right's frame for
    every point in img_left's frame.

    Args:
        img_left (np.ndarray): left image (HxW)
        img_right (np.ndarray): right image (HxW)
        F (np.ndarray): fundamental matrix mapping points in img_left to lines
            in img_right
        w (int): window size in pixels (odd number)
        img_left_mask (np.ndarray): boolean mask that is True in the foreground
            and False in the background. Only compute point correspondences in
            the foreground.

    Returns:
        np.ndarray: point correspondences in the right image for every point in 
            the left image (HxWx2)
    """

    height, width = img_left.shape
    assert img_left.shape == img_right.shape

    # Ensure window size is an odd number
    if w % 2 == 0:
        w += 1
    pt_corr = np.zeros((height, width, 2)).astype(int)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if img_left_mask[i, j]:

                epiline = point_to_epiline(np.array([[j, i]]), F)

                u = epiline[0][0] / epiline[0][2]
                v = epiline[0][1] / epiline[0][2]
                line = utils.compute_line_coordinates(np.array([u, v, 1]).T, width)
                corr = None
                max_ncc = float('-inf')

                r_start, r_end = max(0, i - w // 2), min(height, i + w // 2 + 1)
                c_start, c_end = max(0, j - w // 2), min(width, j + w // 2 + 1)

                window_l = img_left[r_start:r_end, c_start:c_end]

                for row, col in line:
                    if 0 <= col < width and 0 <= row < height:

                        right_x_min = max(col - w // 2, 0)
                        right_x_max = min(col + w // 2 + 1, width)
                        right_y_min = max(row - w // 2, 0)
                        right_y_max = min(row + w // 2 + 1, height)

                        window_r = img_right[right_y_min:right_y_max, right_x_min:right_x_max]

                        # Ensure windows are of the same size
                        min_height = min(window_l.shape[0], window_r.shape[0])
                        min_width = min(window_l.shape[1], window_r.shape[1])
                        window_l = window_l[:min_height, :min_width]
                        window_r = window_r[:min_height, :min_width]

                        ncc = similarity_metric(window_l, window_r)
                        if ncc > max_ncc:
                            max_ncc = ncc
                            corr = (row, col)

                if corr is not None:
                    pt_corr[i, j, :] = corr
            else:
                pt_corr[i, j, :] = (0, 0)

    return pt_corr
