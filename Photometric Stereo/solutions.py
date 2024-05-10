import math
from typing import List, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------
# Photometric Stereo
# --------------------------------------------------------------------------

def find_sphere(
    img: np.ndarray,
) -> Union[Tuple[Tuple[float, float], float], Tuple[None, None]]:
    """
    Locate the center of a (single) sphere in an image and compute its radius.

    Args:
        img (np.ndarray): Input image.

    Returns:
        Union[Tuple[Tuple[float, float], float], Tuple[None, None]]: The center and radius of the sphere, where the center
            is a 2 element tuple (x, y) representing the indices of the sphere center, and radius is the pixel radius of
            of the sphere. Return (None, None) if no sphere is located in the image.
    """

    sphere_pix = np.transpose(np.nonzero(img))
    if sphere_pix.size == 0:
        return None, None
    min_index_0, max_index_0 = np.argmin(sphere_pix[:, 0]), np.argmax(sphere_pix[:, 0])
    min_index_1, max_index_1 = np.argmin(sphere_pix[:, 1]), np.argmax(sphere_pix[:, 1])

    max_x = sphere_pix[max_index_0][0]
    min_x = sphere_pix[min_index_0][0]
    max_y = sphere_pix[max_index_1][1]
    min_y = sphere_pix[min_index_1][1]

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    radius = max_x - center_x
    return (center_x, center_y), radius


def compute_light_directions(
    center: Tuple[float, float], radius: float, imgs: List[np.ndarray]
) -> np.ndarray:
    """
    Compute the direction to the light source.

    Args:
        center (Tuple[float, float]): Set of (x, y) coordinates to the sphere center.
        radius (float): Radius of the circle produced by the sphere on the image plane.
        imgs (List[np.ndarray]): An iterable of images of an object.

    Returns:
        np.ndarray: Nx3 array, where N=len(imgs), containing the normal vector to the brightest surface spot on
            the sphere in each image in imgs.
    """
    norms = []
    for i in range(len(imgs)):
        bright = np.unravel_index(np.argmax(imgs[i]), imgs[i].shape)
        mag = imgs[i][bright[0]][bright[1]]

        y_norm = bright[0] - center[0]
        x_norm = bright[1] - center[1]
        norm = (x_norm, y_norm, math.sqrt(radius ** 2 - x_norm ** 2 - y_norm ** 2)* -1)

        mag_norm = np.linalg.norm(norm)
        scale = mag / mag_norm if mag_norm != 0 else 0

        scale_norm = [j * scale for j in norm]
        norms.append(scale_norm)

    norms = np.array(norms)
    return norms


def compute_mask(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Compute a binary foreground mask for the object.

    Args:
        imgs (List[np.ndarray]): An iterable of images of an object.

    Returns:
        np.ndarray: A binary mask where 1=object, 0=background.
    """
    mask = np.zeros(imgs[0].shape)
    row, col = imgs[0].shape[:2]
    threshold = 3
    print(imgs[0].shape)
    for i in range(row):
        for j in range(col):
            count = 0
            for img in imgs:
                if img[i][j][0] >= threshold:
                    count += 1
            if count >= 3:
                mask[i][j] = 1

    return mask


def compute_normals(
    light_dirs: np.ndarray, imgs: List[np.ndarray], mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the normals and albedo to an object's surface.

    Args:
        light_dirs (np.ndarray): 5x3 lighting direction matrix from
            compute_light_directions.
        imgs (List[np.ndarray]): An iterable of images of an object.
        mask (np.ndarray): Binary foreground mask of an object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The surface normals and albedo of the object.
    """
    col, row = imgs[0].shape[:2]
    normals = np.zeros((col, row, 3))
    albedo = np.zeros((col, row))

    for i in range(row):
        for j in range(col):
            if mask[i][j]:
                intensities = np.array([img[i, j] for img in imgs])

                normal = np.linalg.lstsq(light_dirs, intensities, rcond=None)[0]

                norm = np.linalg.norm(normal)
                if norm != 0:
                    normal /= norm

                normals[i, j] = normal
                albedo[i, j] = norm

    albedo = albedo / albedo.max()
    return normals, albedo


def scene_relighting(
    mask: np.ndarray,
    normals: np.ndarray,
    albedo_img: np.ndarray,
    pointcloud: np.ndarray,
    point_source_xyz: np.ndarray,
    point_source_intensity: float,
) -> np.ndarray:
    """
    Compute the surface radiance at each point on the vase under the given
    illumination. Assume the vase is lambertian.

    Args:
        mask (np.ndarray): A binary mask where 1=object, 0=background.
        normals (np.ndarray): The normal vector at each point on the object image.
        albedo_img (np.ndarray): The albedo image of the object.
        pointcloud (np.ndarray): the xyz location of each point in the image.
        point_source_xyz (np.ndarray): The xyz location of the point source illuminating the object.
        point_source_intensity (np.ndarray): The intensity location of the point source illuminating the object.

    Returns:
        np.ndarray: A relit image of the vase.
    """
    col, row = albedo_img.shape[:2]
    lit_img = np.zeros_like(albedo_img)

    for i in range(row):
        for j in range(col):
            if mask[i][j]:
                s = [point_source_xyz[0] - pointcloud[i,j][0], point_source_xyz[1] - pointcloud[i,j][1],
                     point_source_xyz[2] - pointcloud[i,j][2]]
                norm_s = np.linalg.norm(s)
                unit_s = np.array([val / norm_s for val in s])
                norm = normals[i][j]
                dot = np.sum(norm * unit_s)
                if dot < 0:
                    lit_img[i][j] = 0
                    continue
                r = np.sqrt(np.sum(pointcloud[i][j]-point_source_xyz) ** 2)

                L = (albedo_img[i][j] / np.pi) * (point_source_intensity / (r ** 2)) * dot

                lit_img[i][j] = L

    return lit_img
