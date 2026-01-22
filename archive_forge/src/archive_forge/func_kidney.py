import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def kidney():
    """Mouse kidney tissue.

    This biological tissue on a pre-prepared slide was imaged with confocal
    fluorescence microscopy (Nikon C1 inverted microscope).
    Image shape is (16, 512, 512, 3). That is 512x512 pixels in X-Y,
    16 image slices in Z, and 3 color channels
    (emission wavelengths 450nm, 515nm, and 605nm, respectively).
    Real-space voxel size is 1.24 microns in X-Y, and 1.25 microns in Z.
    Data type is unsigned 16-bit integers.

    Notes
    -----
    This image was acquired by Genevieve Buckley at Monasoh Micro Imaging in
    2018.
    License: CC0

    Returns
    -------
    kidney : (16, 512, 512, 3) uint16 ndarray
        Kidney 3D multichannel image.
    """
    return _load('data/kidney.tif')