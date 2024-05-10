from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
from PIL import Image

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def get_data_path(filename):
    """
    Return the path to a data file.
    """
    p = DATA_DIR / Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())

def get_result_path(filename):
    """
    Return the path to a data file.
    """
    p = RESULTS_DIR / Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())

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
        img = img.astype(np.float64) / 255

    alpha = img[:,:,-1]
    img = img[:,:,:-1]

    return img, alpha

def imshow(img, title=None, flag=cv2.COLOR_BGR2RGB):
    """
    Display the image.
    """
    plt.figure()
    if flag is not None:
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

def get_images(path):
    """
    Get all images in a directory.

    return: list of image paths
    """
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return sorted([f.resolve() for f in Path(path).rglob('*.png')])

def choose_target(img):
    # Pick an initial tracking location
    cv2.imshow('Image', img)
    print('===========')
    print('Drag a rectangle around the tracking target: ')
    rect = cv2.selectROI('Image', img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # To make things easier, let's make the height and width all odd
    rect = list(rect)
    if rect[2] % 2 == 0:
        rect[2] += 1
    if rect[3] % 2 == 0:
        rect[3] += 1

    print(f'[xmin ymin width height] = {rect}')
    print('===========')
    return rect

def annotate_img(img, rect, path):
    # Annotate the image with the tracking rectangle
    img_annotated = img.copy()
    xmin, ymin, width, height = rect
    xmax = xmin + width
    ymax = ymin + height
    cv2.rectangle(img_annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    # Save the image
    imwrite(get_result_path(f'{path}'), img_annotated)
    return img_annotated

def read_gif(path):
    """
    Read a gif file and returns a list of frames
    """
    imgs = imageio.mimread(path)
    img_list = [(cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32) / 255) for img in imgs]
    return img_list

def save_gif(path, frames, fps=2):
    """
    Save a list of frames as a gif file
    """
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames]
    imageio.mimsave(path, imgs, duration=1000./fps)

def show_gif(frames, fps=2):
    """
    Display a gif file
    """
    fig = plt.figure()
    images = []
    for i in range(len(frames)):
        images.append([plt.imshow(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), animated=True)])
    ani = animation.ArtistAnimation(fig, images, interval=1000./fps, blit=True, repeat=True, repeat_delay=0)
    plt.show()

def draw_flow_arrows(img, flow, step=8, scale=4, L=4):
    """
    Modified from: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    img: Frame to draw the flow
    flow: Flow vectors
    step: Number of pixels each vector represents
    scale: Scale factor for the flow vectors
    L: upsample image by L for prettier visualizations

    return: annotated image
    """

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T * scale

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_result = img.copy()
    img_result = cv2.resize(img_result, (img_result.shape[1] * L, img_result.shape[0] * L),
                            interpolation=cv2.INTER_NEAREST)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(img_result, (x1*L, y1*L), (x2*L, y2*L), (0, 0, 255), L // 2)

    return img_result
