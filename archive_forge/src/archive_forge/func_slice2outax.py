import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def slice2outax(ndim, sliceobj):
    """Matching output axes for input array ndim `ndim` and slice `sliceobj`

    Parameters
    ----------
    ndim : int
        number of axes in input array
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``

    Returns
    -------
    out_ax_inds : tuple
        Say ``A` is a (pretend) input array of `ndim` dimensions. Say ``B =
        A[sliceobj]``.  `out_ax_inds` has one value per axis in ``A`` giving
        corresponding axis in ``B``.
    """
    sliceobj = canonical_slicers(sliceobj, [1] * ndim, check_inds=False)
    out_ax_no = 0
    out_ax_inds = []
    for obj in sliceobj:
        if isinstance(obj, Integral):
            out_ax_inds.append(None)
            continue
        if obj is not None:
            out_ax_inds.append(out_ax_no)
        out_ax_no += 1
    return tuple(out_ax_inds)