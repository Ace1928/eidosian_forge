import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def checkerboard():
    """Checkerboard image.

    Checkerboards are often used in image calibration, since the
    corner-points are easy to locate.  Because of the many parallel
    edges, they also visualise distortions particularly well.

    Returns
    -------
    checkerboard : (200, 200) uint8 ndarray
        Checkerboard image.
    """
    return _load('data/chessboard_GRAY.png')