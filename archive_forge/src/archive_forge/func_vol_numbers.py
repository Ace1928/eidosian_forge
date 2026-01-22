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
def vol_numbers(slice_nos):
    """Calculate volume numbers inferred from slice numbers `slice_nos`

    The volume number for each slice is the number of times this slice number
    has occurred previously in the `slice_nos` sequence

    Parameters
    ----------
    slice_nos : sequence
        Sequence of slice numbers, e.g. ``[1, 2, 3, 4, 1, 2, 3, 4]``.

    Returns
    -------
    vol_nos : list
        A list, the same length of `slice_nos` giving the volume number for
        each corresponding slice number.
    """
    counter = {}
    vol_nos = []
    for s_no in slice_nos:
        count = counter.setdefault(s_no, 0)
        vol_nos.append(count)
        counter[s_no] += 1
    return vol_nos