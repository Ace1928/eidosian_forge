import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_assign_fields_by_name_dispatcher)
def assign_fields_by_name(dst, src, zero_unassigned=True):
    """
    Assigns values from one structured array to another by field name.

    Normally in numpy >= 1.14, assignment of one structured array to another
    copies fields "by position", meaning that the first field from the src is
    copied to the first field of the dst, and so on, regardless of field name.

    This function instead copies "by field name", such that fields in the dst
    are assigned from the identically named field in the src. This applies
    recursively for nested structures. This is how structure assignment worked
    in numpy >= 1.6 to <= 1.13.

    Parameters
    ----------
    dst : ndarray
    src : ndarray
        The source and destination arrays during assignment.
    zero_unassigned : bool, optional
        If True, fields in the dst for which there was no matching
        field in the src are filled with the value 0 (zero). This
        was the behavior of numpy <= 1.13. If False, those fields
        are not modified.
    """
    if dst.dtype.names is None:
        dst[...] = src
        return
    for name in dst.dtype.names:
        if name not in src.dtype.names:
            if zero_unassigned:
                dst[name] = 0
        else:
            assign_fields_by_name(dst[name], src[name], zero_unassigned)