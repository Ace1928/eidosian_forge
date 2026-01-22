import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def lily():
    """Lily of the valley plant stem.

    This plant stem on a pre-prepared slide was imaged with confocal
    fluorescence microscopy (Nikon C1 inverted microscope).
    Image shape is (922, 922, 4). That is 922x922 pixels in X-Y,
    with 4 color channels.
    Real-space voxel size is 1.24 microns in X-Y.
    Data type is unsigned 16-bit integers.

    Notes
    -----
    This image was acquired by Genevieve Buckley at Monasoh Micro Imaging in
    2018.
    License: CC0

    Returns
    -------
    lily : (922, 922, 4) uint16 ndarray
        Lily 2D multichannel image.
    """
    return _load('data/lily.tif')