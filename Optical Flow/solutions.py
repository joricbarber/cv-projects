import numpy as np
from scipy import signal


# --------------------------------------------------------------------------
# Optical flow using Lucas-Kanade
# --------------------------------------------------------------------------
def computeFlow(img1: np.ndarray, img2: np.ndarray, window_size: int) -> np.ndarray:
    """
    Use Lucas-Kanade method to compute the optical flow between two images.

    At each pixel, compute the optical flow (u, v) between the two images. u
    is the flow in the x (column) direction, and v is the flow in the y (row)
    direction.

    Args:
        img1 (np.ndarray): the first image (HxW)
        img2 (np.ndarray): the second image (HxW)
        window_size (int): size of the window within which optical flow is constant

    Returns:
        np.ndarray: calculated optical flow (u, v) at each pixel (HxWx2)
    """

    Ix = np.zeros_like(img1, dtype=float)
    Iy = np.zeros_like(img1, dtype=float)
    It = img2 - img1

    # Compute gradients for all pixels
    for k in range(img1.shape[0] - 1):
        for l in range(img1.shape[1] - 1):
            Ix[k, l] = 0.25 * (img1[k + 1, l] + img1[k + 1, l + 1] + img2[k + 1, l] + img2[k + 1, l + 1]) \
                       - 0.25 * (img1[k, l] + img1[k, l + 1] + img2[k, l] + img2[k, l + 1])
            Iy[k, l] = 0.25 * (img1[k, l + 1] + img1[k + 1, l + 1] + img2[k, l + 1] + img2[k + 1, l + 1]) \
                       - 0.25 * (img1[k, l] + img1[k + 1, l] + img2[k, l] + img2[k + 1, l])

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    half_w = window_size // 2

    for i in range(half_w, img1.shape[0] - half_w, window_size):
        for j in range(half_w, img1.shape[1] - half_w, window_size):

            Ix_win = Ix[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].flatten()
            Iy_win = Iy[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].flatten()
            It_win = It[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].flatten()

            A = np.vstack((Ix_win, Iy_win)).T
            B = -It_win
            if np.linalg.det(A.T @ A) != 0:
                optical_flow = np.linalg.pinv(A.T @ A) @ A.T @ B
                for s in range(i - half_w, i + half_w + 1):
                    for t in range(j - half_w, j + half_w + 1):
                        u[s, t] = optical_flow[0]
                        v[s, t] = optical_flow[1]

    flows = np.stack((v, u), axis=2)

    return flows
