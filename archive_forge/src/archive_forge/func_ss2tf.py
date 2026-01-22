import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def ss2tf(A, B, C, D, input=0):
    """State-space to transfer function.

    A, B, C, D defines a linear state-space system with `p` inputs,
    `q` outputs, and `n` state variables.

    Parameters
    ----------
    A : array_like
        State (or system) matrix of shape ``(n, n)``
    B : array_like
        Input matrix of shape ``(n, p)``
    C : array_like
        Output matrix of shape ``(q, n)``
    D : array_like
        Feedthrough (or feedforward) matrix of shape ``(q, p)``
    input : int, optional
        For multiple-input systems, the index of the input to use.

    Returns
    -------
    num : 2-D ndarray
        Numerator(s) of the resulting transfer function(s). `num` has one row
        for each of the system's outputs. Each row is a sequence representation
        of the numerator polynomial.
    den : 1-D ndarray
        Denominator of the resulting transfer function(s). `den` is a sequence
        representation of the denominator polynomial.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.ss2tf

    """
    A, B, C, D = abcd_normalize(A, B, C, D)
    nout, nin = D.shape
    if input >= nin:
        raise ValueError('System does not have the input specified.')
    B = B[:, input:input + 1]
    D = D[:, input:input + 1]
    try:
        den = poly(A)
    except ValueError:
        den = 1
    if prod(B.shape) == 0 and prod(C.shape) == 0:
        num = cupy.ravel(D)
        if prod(D.shape) == 0 and prod(A.shape) == 0:
            den = []
        return (num, den)
    num_states = A.shape[0]
    type_test = A[:, 0] + B[:, 0] + C[0, :] + D + 0.0
    num = cupy.empty((nout, num_states + 1), type_test.dtype)
    for k in range(nout):
        Ck = cupy.atleast_2d(C[k, :])
        num[k] = poly(A - B @ Ck) + (D[k] - 1) * den
    return (num, den)