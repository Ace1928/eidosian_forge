import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def ir2tf(imp_resp, shape, dim=None, is_real=True):
    """Compute the transfer function of an impulse response (IR).

    This function makes the necessary correct zero-padding, zero
    convention, correct fft2, etc... to compute the transfer function
    of IR. To use with unitary Fourier transform for the signal (ufftn
    or equivalent).

    Parameters
    ----------
    imp_resp : ndarray
        The impulse responses.
    shape : tuple of int
        A tuple of integer corresponding to the target shape of the
        transfer function.
    dim : int, optional
        The last axis along which to compute the transform. All
        axes by default.
    is_real : boolean, optional
       If True (default), imp_resp is supposed real and the Hermitian property
       is used with rfftn Fourier transform.

    Returns
    -------
    y : complex ndarray
       The transfer function of shape ``shape``.

    See Also
    --------
    ufftn, uifftn, urfftn, uirfftn

    Examples
    --------
    >>> np.all(np.array([[4, 0], [0, 0]]) == ir2tf(np.ones((2, 2)), (2, 2)))
    True
    >>> ir2tf(np.ones((2, 2)), (512, 512)).shape == (512, 257)
    True
    >>> ir2tf(np.ones((2, 2)), (512, 512), is_real=False).shape == (512, 512)
    True

    Notes
    -----
    The input array can be composed of multiple-dimensional IR with
    an arbitrary number of IR. The individual IR must be accessed
    through the first axes. The last ``dim`` axes contain the space
    definition.
    """
    if not dim:
        dim = imp_resp.ndim
    irpadded_dtype = _supported_float_type(imp_resp.dtype)
    irpadded = np.zeros(shape, dtype=irpadded_dtype)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= imp_resp.ndim - dim:
            irpadded = np.roll(irpadded, shift=-int(np.floor(axis_size / 2)), axis=axis)
    func = fft.rfftn if is_real else fft.fftn
    out = func(irpadded, axes=range(-dim, 0))
    cplx_dtype = np.promote_types(irpadded_dtype, np.complex64)
    return out.astype(cplx_dtype, copy=False)