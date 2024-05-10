from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.draw
import scipy
from typing import Tuple

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def extract_intrinsics(P: np.ndarray) -> np.ndarray:
    """
    Extract the intrinsic matrix, K, from the projection matrix, P.

    Args:
        P (np.ndarray): projection matrix (3x4)

    Returns:
        np.ndarray: intrinsic matrix (3x3)
    """
    K = scipy.linalg.cholesky(P[:,:3].T @ P[:,:3], lower=False)
    K /= K[-1,-1]
    return K


def extract_RT(E: np.ndarray, 
               K: np.ndarray, 
               uv1: np.ndarray, 
               uv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the rotation matrix and translation vector from the essential
    matrix.

    Args:
        E (np.ndarray): essential matrix (3x3)
        K (np.ndarray): intrinsic matrix (3x3)
        uv1 (np.ndarray): points in left image (Nx2)
        uv2 (np.ndarray): points in right image (Nx2)

    Returns:
        Tuple[np.ndarray, np.ndarray]: rotation matrix (3x3), 
            translation vector (3x1)
    """
    _, R, t, _ = cv2.recoverPose(E, uv1, uv2, K)
    assert np.allclose(np.linalg.norm(t), 1)
    return R, t


def sift_matches(img1, img2):
    """
    Obtain point correspondences using SIFT features.
    img1: First image
    img2: Second image

    src_pts: Nx2 points in img1
    dest_pts: Nx2 corresponding points in img2
    """

    if img1.dtype == np.float64 or img1.dtype == np.float32:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype == np.float64 or img2.dtype == np.float32:
        img2 = (img2 * 255).astype(np.uint8)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    src_pts = np.asarray(
        [kp1[good[i][0].queryIdx].pt for i in range(len(good))])
    dest_pts = np.asarray(
        [kp2[good[i][0].trainIdx].pt for i in range(len(good))])

    return src_pts, dest_pts


def show_correspondences(src_img, dest_img, src_pts, dest_pts, title=None):
    """
    Visualize correspondences between two images by plotting both images
    side-by-side and drawing lines between each point correspondence.

    src_img: source image
    dest_img: destination image
    src_pts: point correspondences in the source image (Nx2)
    dest_pts: point correspondences in the destination image (Nx2)
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2

    N = src_pts.shape[0]

    fig, ax = plt.subplots()
    plt.axis("off")

    ax.imshow(np.hstack((src_img, dest_img)))
    t = src_img.shape[1]
    for i in range(N):
        # Draw line
        xs = src_pts[i,:]
        xd = dest_pts[i,:]
        ax.plot([xs[0], xd[0]+t], [xs[1], xd[1]], 'r-', linewidth=0.75)

    if title is not None:
        plt.title(title)

    return fig


def get_data_path(filename):
    """
    Return the path to a data file.
    """
    return str((DATA_DIR / filename).resolve())


def get_result_path(filename):
    """
    Return the path to a data file.
    """
    return str((RESULTS_DIR / filename).resolve())


def imread(path, flag=cv2.IMREAD_COLOR, rgb=False, normalize=False):
    """
    Read an image from a file.

    path: Image path
    flag: flag passed to cv2.imread
    normalize: normalize the values to [0, 1]
    rgb: convert BGR to RGB
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(str(path), flag)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255
    return img


def imread_alpha(path, normalize=False):
    """
    Read an image from a file.
    Use this function when the image contains an alpha channel. That channel
    is returned separately.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if normalize:
        img = img.astype(np.float32) / 255

    alpha = img[:,:,-1]
    img = img[:,:,:-1]

    return img, alpha


def imshow(img, title=None, flag=cv2.COLOR_BGR2RGB):
    """
    Display the image.
    """
    plt.figure()
    if flag is not None:
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        img = cv2.cvtColor(img, flag)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def imwrite(path, img, flag=None):
    """
    Write the image to a file.
    """
    assert type(img) == np.ndarray
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    cv2.imwrite(str(path), img)


def show_correspondences(src_img, dest_img, src_pts, dest_pts, title=None):
    """
    Visualize correspondences between two images by plotting both images
    side-by-side and drawing lines between each point correspondence.

    src_img: source image
    dest_img: destination image
    src_pts: point correspondences in the source image (Nx2)
    dest_pts: point correspondences in the destination image (Nx2)
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2

    N = src_pts.shape[0]

    fig, ax = plt.subplots()
    plt.axis("off")

    ax.imshow(np.hstack((src_img, dest_img)))
    t = src_img.shape[1]
    for i in range(N):
        # Draw line
        xs = src_pts[i,:]
        xd = dest_pts[i,:]
        ax.plot([xs[0], xd[0]+t], [xs[1], xd[1]], 'r-', linewidth=0.75)

    if title is not None:
        plt.title(title)

    return fig

def draw_line(ax, line, img_size):
    width = img_size[1]
    X = np.asarray([
        [0, -line[2]/line[1]],
        [width, -(line[2] + line[0]*width)/line[1]]
    ])
    ax.plot(X[:,0], X[:,1], color="#009dff", linewidth=2)


def compute_line_coordinates(line: np.ndarray, img_width: int) -> np.ndarray:
    """
    Compute the pixel coordinates that a line passes through in the image.

    Args:
        line (np.ndarray): 3x1 vector of the line in homogeneous coordinates
        img_width (int): image width in pixels

    Returns:
        np.ndarray: Nx2 array of pixel coordinates in the image in (row, column)
            order
    """

    X0 = np.round(np.asarray([0, -line[2] / line[1]])).astype(int)
    X1 = np.round(np.asarray(
        [img_width, -(line[2] + line[0]*img_width)/line[1]])).astype(int)
    line_points = skimage.draw.line(X0[1], X0[0], X1[1], X1[0])
    line_points = np.stack(line_points).T # N x 2
    return line_points


def triangulate_point_cloud(img_left):
    # Load intrinsics
    K = np.load(get_result_path("K.npy")) # type:np.ndarray

    # Load pose
    pose = np.load(get_result_path("challenge1b.npz"))
    R, t = pose["R"], pose["t"]

    # Load point correspondences
    D = np.load(get_result_path("challenge1c.npz"))
    pts_in_img_right = D["pts_in_img_right"]

    r_left, c_left = np.meshgrid(
        np.arange(img_left.shape[0]),
        np.arange(img_left.shape[1]), indexing="ij")
    pts_in_img_left = np.stack((r_left.ravel(), c_left.ravel())) # row, column
    pts_in_img_right = np.transpose(pts_in_img_right, (2, 0, 1)).reshape(2, -1) # row, column

    P_left = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P_right = K @ np.hstack((R, t))
    X_world = cv2.triangulatePoints(
        P_left, P_right,
        np.flipud(pts_in_img_left.astype(np.float32)), # flip row, column -> x, y
        np.flipud(pts_in_img_right.astype(np.float32)))
    X_world = X_world[:3] / X_world[3]

    return X_world, pts_in_img_left, pts_in_img_right
