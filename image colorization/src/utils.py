from skimage import io, color
from skimage.transform import rescale, resize
import numpy as np


def read_image(filename, size=(256, 256), training=False):
    img = io.imread(filename)
    real_size = img.shape
    if img.shape!=size and not training:
        img     = resize(img, size, anti_aliasing=False)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img, real_size[:2]


def cvt2Lab(image):
    Lab = color.rgb2lab(image)
    return Lab[:, :, 0], Lab[:, :, 1:]  # L, ab


def cvt2rgb(image):
    return color.lab2rgb(image)


def upsample(image):
    return rescale(image, 4, mode='constant', order=3)