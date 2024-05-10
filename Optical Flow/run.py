import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import solutions as solutions
import utils


def run():

    np.set_printoptions(precision=4, suppress=True)

    fns = {
        "still": still,
        "video": video,
    }
    if sys.argv[1] == "-h":
        print("Available functions:")
        for s in fns.keys():
            print("  %s" % s)
    elif sys.argv[1] == "all":
        for fn in fns.values():
            fn()
    else:
        for fn_name in sys.argv[1:]:
            fns[fn_name]()


#--------------------------------------------------------------------------
# Optical flow using Lucas-Kanade
#--------------------------------------------------------------------------
def still():
    img_list = ['rubic1', 'rubic2']

    # Load a pair of consecutive images
    img1 = utils.imread(utils.get_data_path(f'rubic/{img_list[0]}.png'), normalize=True)
    img2 = utils.imread(utils.get_data_path(f'rubic/{img_list[1]}.png'), normalize=True)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the optical flow
    flow = solutions.computeFlow(img1_gray, img2_gray, window_size=21)

    # Draw the optical flow field on the first image of the pair
    needle = utils.draw_flow_arrows(img1, flow, step=16, scale=8, L=4)

    # Save the optical flow field as a gif
    utils.imwrite(utils.get_result_path('rubic.png'), needle)
    utils.imshow(needle)


def video():
    frames = utils.read_gif(utils.get_data_path('traffic/traffic.gif'))

    needles = []
    for i in range(len(frames)-1):
        print("Frame %d / %d..." % (i+1, len(frames) - 1), end="\r")

        img1_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        # Compute the optical flow of a pair of consecutive images
        flow = solutions.computeFlow(img1_gray, img2_gray, window_size=35)

        # Draw the optical flow field on the first image of the pair
        needle = utils.draw_flow_arrows(frames[i], flow, step=16, scale=16, L=2)
        needles.append(needle)

    # Save the optical flow field as a gif
    utils.save_gif(utils.get_result_path('traffic.gif'), needles, fps=2)
    utils.show_gif(needles, fps=2)


if __name__ == "__main__":
    run()
