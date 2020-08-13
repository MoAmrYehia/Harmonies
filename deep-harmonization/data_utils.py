import numpy as np

from imageio import imread

def read_img(fn):
    return np.transpose(imread(fn).astype(np.float32), (2, 0, 1))
