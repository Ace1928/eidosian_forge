import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def microaneurysms():
    """Gray-level "microaneurysms" image.

    Detail from an image of the retina (green channel).
    The image is a crop of image 07_dr.JPG from the
    High-Resolution Fundus (HRF) Image Database:
    https://www5.cs.fau.de/research/data/fundus-images/

    Notes
    -----
    No copyright restrictions. CC0 given by owner (Andreas Maier).

    Returns
    -------
    microaneurysms : (102, 102) uint8 ndarray
        Retina image with lesions.

    References
    ----------
    .. [1] Budai, A., Bock, R, Maier, A., Hornegger, J.,
           Michelson, G. (2013).  Robust Vessel Segmentation in Fundus
           Images. International Journal of Biomedical Imaging, vol. 2013,
           2013.
           :DOI:`10.1155/2013/154860`
    """
    return _load('data/microaneurysms.png')