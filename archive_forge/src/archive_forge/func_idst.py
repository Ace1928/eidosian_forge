import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
@_fft._implements(_fft._scipy_fft.idst)
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """Return the Inverse Discrete Sine Transform of an array, x.

    Parameters
    ----------
    x : cupy.ndarray
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    idst : cupy.ndarray of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.idst`

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see :func:`scipy.fft.dst`.
    """
    if x.dtype.kind == 'c':
        out = idst(x.real, type, n, axis, norm, overwrite_x)
        out = out + 1j * idst(x.imag, type, n, axis, norm, overwrite_x)
        return out
    x = _promote_dtype(x)
    if type == 2:
        return _dct_or_dst_type3(x, n=n, axis=axis, norm=norm, forward=False, dst=True)
    elif type == 3:
        return _dct_or_dst_type2(x, n=n, axis=axis, norm=norm, forward=False, dst=True)
    elif type in [1, 4]:
        raise NotImplementedError('Only DST-II and DST-III have been implemented.')
    else:
        raise ValueError('invalid DST type')