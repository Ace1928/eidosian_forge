import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def lfw_subset():
    """Subset of data from the LFW dataset.

    This database is a subset of the LFW database containing:

    * 100 faces
    * 100 non-faces

    The full dataset is available at [2]_.

    Returns
    -------
    images : (200, 25, 25) uint8 ndarray
        100 first images are faces and subsequent 100 are non-faces.

    Notes
    -----
    The faces were randomly selected from the LFW dataset and the non-faces
    were extracted from the background of the same dataset. The cropped ROIs
    have been resized to a 25 x 25 pixels.

    References
    ----------
    .. [1] Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012).
           Learning to align from scratch. In Advances in Neural Information
           Processing Systems (pp. 764-772).
    .. [2] http://vis-www.cs.umass.edu/lfw/

    """
    return np.load(_fetch('data/lfw_subset.npy'))