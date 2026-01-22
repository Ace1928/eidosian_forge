import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def eagle():
    """A golden eagle.

    Suitable for examples on segmentation, Hough transforms, and corner
    detection.

    Notes
    -----
    No copyright restrictions. CC0 by the photographer (Dayane Machado).

    Returns
    -------
    eagle : (2019, 1826) uint8 ndarray
        Eagle image.
    """
    return _load('data/eagle.png')