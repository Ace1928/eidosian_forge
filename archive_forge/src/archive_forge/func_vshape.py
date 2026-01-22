import os
import platform
import itertools
from .xmltodict import parse as xmltodictparser
import subprocess as sp
import numpy as np
from .edge import canny
from .stpyr import SpatialSteerablePyramid, rolling_window
from .mscn import compute_image_mscn_transform, gen_gauss_window
from .stats import ggd_features, aggd_features, paired_product
def vshape(videodata):
    """Standardizes the input data shape.

    Transforms video data into the standardized shape (T, M, N, C), where
    T is number of frames, M is height, N is width, and C is number of 
    channels.

    Parameters
    ----------
    videodata : ndarray
        Input data of shape (T, M, N, C), (T, M, N), (M, N, C), or (M, N), where
        T is number of frames, M is height, N is width, and C is number of 
        channels.

    Returns
    -------
    videodataout : ndarray
        Standardized version of videodata, shape (T, M, N, C)

    """
    if not isinstance(videodata, np.ndarray):
        videodata = np.array(videodata)
    if len(videodata.shape) == 2:
        a, b = videodata.shape
        return videodata.reshape(1, a, b, 1)
    elif len(videodata.shape) == 3:
        a, b, c = videodata.shape
        if c in [1, 2, 3, 4]:
            return videodata.reshape(1, a, b, c)
        else:
            return videodata.reshape(a, b, c, 1)
    elif len(videodata.shape) == 4:
        return videodata
    else:
        raise ValueError('Improper data input')