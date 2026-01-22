import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def check_shape_equality(*images):
    """Check that all images have the same shape"""
    image0 = images[0]
    if not all((image0.shape == image.shape for image in images[1:])):
        raise ValueError('Input images must have the same dimensions.')
    return