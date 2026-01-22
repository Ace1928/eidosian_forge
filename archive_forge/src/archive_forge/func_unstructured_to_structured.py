import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_unstructured_to_structured_dispatcher)
def unstructured_to_structured(arr, dtype=None, names=None, align=False, copy=False, casting='unsafe'):
    """
    Converts an n-D unstructured array into an (n-1)-D structured array.

    The last dimension of the input array is converted into a structure, with
    number of field-elements equal to the size of the last dimension of the
    input array. By default all output fields have the input array's dtype, but
    an output structured dtype with an equal number of fields-elements can be
    supplied instead.

    Nested fields, as well as each element of any subarray fields, all count
    towards the number of field-elements.

    Parameters
    ----------
    arr : ndarray
       Unstructured array or dtype to convert.
    dtype : dtype, optional
       The structured dtype of the output array
    names : list of strings, optional
       If dtype is not supplied, this specifies the field names for the output
       dtype, in order. The field dtypes will be the same as the input array.
    align : boolean, optional
       Whether to create an aligned memory layout.
    copy : bool, optional
        See copy argument to `numpy.ndarray.astype`. If true, always return a
        copy. If false, and `dtype` requirements are satisfied, a view is
        returned.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `numpy.ndarray.astype`. Controls what kind of
        data casting may occur.

    Returns
    -------
    structured : ndarray
       Structured array with fewer dimensions.

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> dt = np.dtype([('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
    >>> a = np.arange(20).reshape((4,5))
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    >>> rfn.unstructured_to_structured(a, dt)
    array([( 0, ( 1.,  2), [ 3.,  4.]), ( 5, ( 6.,  7), [ 8.,  9.]),
           (10, (11., 12), [13., 14.]), (15, (16., 17), [18., 19.])],
          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])

    """
    if arr.shape == ():
        raise ValueError('arr must have at least one dimension')
    n_elem = arr.shape[-1]
    if n_elem == 0:
        raise NotImplementedError('last axis with size 0 is not supported')
    if dtype is None:
        if names is None:
            names = ['f{}'.format(n) for n in range(n_elem)]
        out_dtype = np.dtype([(n, arr.dtype) for n in names], align=align)
        fields = _get_fields_and_offsets(out_dtype)
        dts, counts, offsets = zip(*fields)
    else:
        if names is not None:
            raise ValueError("don't supply both dtype and names")
        dtype = np.dtype(dtype)
        fields = _get_fields_and_offsets(dtype)
        if len(fields) == 0:
            dts, counts, offsets = ([], [], [])
        else:
            dts, counts, offsets = zip(*fields)
        if n_elem != sum(counts):
            raise ValueError('The length of the last dimension of arr must be equal to the number of fields in dtype')
        out_dtype = dtype
        if align and (not out_dtype.isalignedstruct):
            raise ValueError('align was True but dtype is not aligned')
    names = ['f{}'.format(n) for n in range(len(fields))]
    packed_fields = np.dtype({'names': names, 'formats': [(arr.dtype, dt.shape) for dt in dts]})
    arr = np.ascontiguousarray(arr).view(packed_fields)
    flattened_fields = np.dtype({'names': names, 'formats': dts, 'offsets': offsets, 'itemsize': out_dtype.itemsize})
    arr = arr.astype(flattened_fields, copy=copy, casting=casting)
    return arr.view(out_dtype)[..., 0]