import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import solutions as solutions
import utils

left_img_name = "face-left.png"
right_img_name = "face-right.png"
left_mask_name = "face-left-mask.png"

def run():
    np.set_printoptions(precision=4, suppress=True)

    fns = {
        "calibrate": calibrate,
        "find_pose": find_pose,
        "correspondence": correspondence,
        "show_point_cloud": show_point_cloud
    }
    if len(sys.argv) < 2 or sys.argv[1] == "-h":
        print("Available functions:")
        for s in fns.keys():
            print("  %s" % s)
    elif sys.argv[1] == "all":
        for fn in fns.values():
            fn()
    elif sys.argv[1] not in fns.keys():
        print("Function not available: %s" % sys.argv[1])
    else:
        for fn_name in sys.argv[1:]:
            fns[fn_name]()


def calibrate():
    # Load calibration points
    D = scipy.io.loadmat(utils.get_data_path("point_correspondences.mat"))
    X_world = D["X_world"] # type:np.ndarray
    uv = D["uv"] # type:np.ndarray

    # Estimate projection matrix
    P = solutions.estimate_P(uv, X_world)
    reproj_error = solutions.reprojection_error(uv, X_world, P)

    print("Reprojection Error: %.2fpx" % reproj_error)
    print()

    # Extract the intrinsics
    K = utils.extract_intrinsics(P)
    print("Intrinsics: ")
    print(K)

    np.save(utils.get_result_path("K"), K)

def find_pose():
    def key_handler(event):
        if event.key == 'escape':
            sys.exit(0)

    # Load intrinsics from above
    K = np.load(utils.get_result_path("K.npy")) # type:np.ndarray

    img_left = utils.imread(utils.get_data_path(left_img_name), rgb=True)
    img_right = utils.imread(utils.get_data_path(right_img_name), rgb=True)
    uv1, uv2 = utils.sift_matches(img_left, img_right)

    F, inlier_idx, residual = solutions.estimate_F_RANSAC(uv1, uv2)
    print("Mean error on inliers: %.2fpx" % residual[inlier_idx].mean())
    print("Number of inliers: %d" % len(inlier_idx))

    E = solutions.estimate_E(F, K)
    R, t = utils.extract_RT(E, K, uv1[inlier_idx,:], uv2[inlier_idx,:])

    np.savez(utils.get_result_path("challenge1b"),
             F=F,
             E=E,
             R=R,
             t=t,
             inlier_idx=inlier_idx,
             uv1=uv1,
             uv2=uv2)

    ## Interactively test the fundamental matrix
    img_left = (img_left.astype(np.float32) / 255)**(1/2.2)
    img_right = (img_right.astype(np.float32) / 255)**(1/2.2)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
    fig.suptitle("Press Escape to Exit")
    fig.canvas.mpl_connect('key_press_event', key_handler)
    fig.tight_layout()
    fig.show()

    while True:
        ax0.clear()
        ax0.imshow(img_left)
        ax0.set_title("Click a point here...")
        ax0.axis("off")

        ax1.clear()
        ax1.imshow(img_right)
        ax1.set_title("...check the epiline")
        ax1.axis("off")

        pt = plt.ginput(1, timeout=10, show_clicks=False)
        pt = np.asarray(pt[0])

        ax0.plot(pt[0], pt[1], 'ro', ms=4)
        line = solutions.point_to_epiline(pt[None,:], F).squeeze()
        utils.draw_line(ax1, line, img_right.shape)
        ax1.set_xlim([0, img_right.shape[1]])
        ax1.set_ylim([img_right.shape[0], 0])

        fig.canvas.draw()
        fig.canvas.flush_events()

def correspondence():
    """
    Compute dense point correspondences using template matching.
    """
    img_left = utils.imread(utils.get_data_path(left_img_name),
                            normalize=True)
    img_right = utils.imread(utils.get_data_path(right_img_name),
                             normalize=True)
    mask = utils.imread(utils.get_data_path(left_mask_name),
                                 cv2.IMREAD_GRAYSCALE) > 128

    # Convert to gray scale
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    D = np.load(utils.get_result_path("challenge1b.npz"))
    F = D["F"] # type:np.ndarray

    pts_in_img_right = solutions.dense_point_correspondences(
        img_left, img_right, F, solutions.window_size, mask)

    np.savez(utils.get_result_path("challenge1c"),
             pts_in_img_right=pts_in_img_right)

def show_point_cloud():
    """
    Show the point cloud in an interactive plot.
    """
    img_left = utils.imread(utils.get_data_path(left_img_name), rgb=True)
    D = np.load(utils.get_result_path("challenge1c.npz"))
    mask = utils.imread(utils.get_data_path(left_mask_name),
                        cv2.IMREAD_GRAYSCALE) > 128

    X_world, pts_in_img_left, pts_in_img_right = \
        utils.triangulate_point_cloud(img_left)

    Z_MIN = 5.5 # Minimum depth
    Z_MAX = 9 # Maximum depth

    good_idx = mask.ravel() & \
        (X_world[2] > Z_MIN) & (X_world[2] < Z_MAX)

    color = img_left[pts_in_img_left[0,:], pts_in_img_right[1,:]]
    X_world = X_world[:,good_idx]
    color = (color[good_idx,:] / 255)**(1/2.2)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X_world[0,:],
               X_world[1,:],
               X_world[2,:],
               s=2, c=color)
    ax.axis("off")
    ax.axis("equal")
    ax.view_init(elev=-62, azim=-88, roll=0)
    fig.tight_layout()
    fig.savefig(utils.get_result_path("point_cloud.png"), dpi=192)
    plt.show()

if __name__ == "__main__":
    run()
