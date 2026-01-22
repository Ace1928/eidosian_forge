import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def get_moment3D_function(img, spacing=(1, 1, 1)):
    slices, rows, cols = img.shape
    Z, Y, X = np.meshgrid(np.linspace(0, slices * spacing[0], slices, endpoint=False), np.linspace(0, rows * spacing[1], rows, endpoint=False), np.linspace(0, cols * spacing[2], cols, endpoint=False), indexing='ij')
    return lambda p, q, r: np.sum(Z ** p * Y ** q * X ** r * img)