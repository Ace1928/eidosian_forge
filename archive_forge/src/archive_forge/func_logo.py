import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def logo():
    """Scikit-image logo, a RGBA image.

    Returns
    -------
    logo : (500, 500, 4) uint8 ndarray
        Logo image.
    """
    return _load('data/logo.png')