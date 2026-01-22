import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_require_fields_dispatcher)
def require_fields(array, required_dtype):
    """
    Casts a structured array to a new dtype using assignment by field-name.

    This function assigns from the old to the new array by name, so the
    value of a field in the output array is the value of the field with the
    same name in the source array. This has the effect of creating a new
    ndarray containing only the fields "required" by the required_dtype.

    If a field name in the required_dtype does not exist in the
    input array, that field is created and set to 0 in the output array.

    Parameters
    ----------
    a : ndarray
       array to cast
    required_dtype : dtype
       datatype for output array

    Returns
    -------
    out : ndarray
        array with the new dtype, with field values copied from the fields in
        the input array with the same name

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.ones(4, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
    >>> rfn.require_fields(a, [('b', 'f4'), ('c', 'u1')])
    array([(1., 1), (1., 1), (1., 1), (1., 1)],
      dtype=[('b', '<f4'), ('c', 'u1')])
    >>> rfn.require_fields(a, [('b', 'f4'), ('newf', 'u1')])
    array([(1., 0), (1., 0), (1., 0), (1., 0)],
      dtype=[('b', '<f4'), ('newf', 'u1')])

    """
    out = np.empty(array.shape, dtype=required_dtype)
    assign_fields_by_name(out, array)
    return out