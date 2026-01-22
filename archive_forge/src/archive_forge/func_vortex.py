import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def vortex():
    """Case B1 image pair from the first PIV challenge.

    Returns
    -------
    image0, image1 : (512, 512) grayscale images
        A pair of images featuring synthetic moving particles.

    Notes
    -----
    This image was licensed as CC0 by its author, Prof. Koji Okamoto, with
    thanks to Prof. Jun Sakakibara, who maintains the PIV Challenge site.

    References
    ----------
    .. [1] Particle Image Velocimetry (PIV) Challenge site
           http://pivchallenge.org
    .. [2] 1st PIV challenge Case B: http://pivchallenge.org/pub/index.html#b
    """
    return (_load('data/pivchallenge-B-B001_1.tif'), _load('data/pivchallenge-B-B001_2.tif'))