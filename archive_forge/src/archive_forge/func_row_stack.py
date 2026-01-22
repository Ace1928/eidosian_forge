import numpy as _np
from ...base import numeric_types, integer_types
from ...util import _sanity_check_params, set_module
from ...util import wrap_np_unary_func, wrap_np_binary_func
from ...context import current_context
from . import _internal as _npi
from . import _api_internal
from ..ndarray import NDArray
@set_module('mxnet.ndarray.numpy')
def row_stack(arrays):
    """Stack arrays in sequence vertically (row wise).
    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.
    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate` and `stack`
    provide more general stacking and concatenation operations.
    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.
    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.
    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.vstack((a, b))
    array([[1., 2., 3.],
            [2., 3., 4.]])
    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[2], [3], [4]])
    >>> np.vstack((a, b))
    array([[1.],
            [2.],
            [3.],
            [2.],
            [3.],
            [4.]])
    """

    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError('expected iterable for arrays but got {}'.format(type(arrays)))
        return [arr for arr in arrays]
    arrays = get_list(arrays)
    return _npi.vstack(*arrays)