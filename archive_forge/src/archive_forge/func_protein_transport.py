import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def protein_transport():
    """Microscopy image sequence with fluorescence tagging of proteins
    re-localizing from the cytoplasmic area to the nuclear envelope.

    Returns
    -------
    protein_transport: (15, 2, 180, 183) uint8 ndarray

    Notes
    -----
    See info under `NPCsingleNucleus.tif` at
    https://gitlab.com/scikit-image/data/-/blob/master/README.md#data.

    """
    return _load('data/protein_transport.tif')