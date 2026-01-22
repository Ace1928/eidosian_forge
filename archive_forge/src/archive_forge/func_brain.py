import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def brain():
    """Subset of data from the University of North Carolina Volume Rendering
    Test Data Set.

    The full dataset is available at [1]_.

    Returns
    -------
    image : (10, 256, 256) uint16 ndarray

    Notes
    -----
    The 3D volume consists of 10 layers from the larger volume.

    References
    ----------
    .. [1] https://graphics.stanford.edu/data/voldata/

    """
    return _load('data/brain.tiff')