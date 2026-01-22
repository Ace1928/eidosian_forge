import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def fallback_svd(a, full_matrices=True, compute_uv=True):
    try:
        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        pass
    from scipy.linalg import svd
    return svd(a, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver='gesvd')