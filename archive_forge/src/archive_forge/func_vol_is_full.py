import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def vol_is_full(slice_nos, slice_max, slice_min=1):
    """Vector with True for slices in complete volume, False otherwise

    Parameters
    ----------
    slice_nos : sequence
        Sequence of slice numbers, e.g. ``[1, 2, 3, 4, 1, 2, 3, 4]``.
    slice_max : int
        Highest slice number for a full slice set.  Slice set will be
        ``range(slice_min, slice_max+1)``.
    slice_min : int, optional
        Lowest slice number for full slice set.  Default is 1.

    Returns
    -------
    is_full : array
        Bool vector with True for slices in full volumes, False for slices in
        partial volumes.  A full volume is a volume with all slices in the
        ``slice set`` as defined above.

    Raises
    ------
    ValueError
        if any value in `slice_nos` is outside slice set indices.
    """
    slice_set = set(range(slice_min, slice_max + 1))
    if not slice_set.issuperset(slice_nos):
        raise ValueError(f'Slice numbers outside inclusive range {slice_min} to {slice_max}')
    vol_nos = np.array(vol_numbers(slice_nos))
    slice_nos = np.asarray(slice_nos)
    is_full = np.ones(slice_nos.shape, dtype=bool)
    for vol_no in set(vol_nos):
        ours = vol_nos == vol_no
        if not set(slice_nos[ours]) == slice_set:
            is_full[ours] = False
    return is_full