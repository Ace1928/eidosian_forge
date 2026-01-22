import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def lsim(system, U, T, X0=None, interp=True):
    """
    Simulate output of a continuous-time linear system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 1: (instance of `lti`)
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)

    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U = 0 or None, a zero input is used.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.  Must be nonnegative, increasing, and equally spaced
    X0 : array_like, optional
        The initial conditions on the state vector (zero by default).
    interp : bool, optional
        Whether to use linear (True, the default) or zero-order-hold (False)
        interpolation for the input array.

    Returns
    -------
    T : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time evolution of the state vector.

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    See Also
    --------
    scipy.signal.lsim

    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_ss()
    T = cupy.atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError('T must be a rank-1 array.')
    A, B, C, D = map(cupy.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_steps = T.size
    if X0 is None:
        X0 = cupy.zeros(n_states, sys.A.dtype)
    xout = cupy.empty((n_steps, n_states), sys.A.dtype)
    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        xout[0] = X0 @ expm(A.T * T[0])
    else:
        raise ValueError('Initial time must be nonnegative')
    no_input = U is None or (isinstance(U, (int, float)) and U == 0.0) or (not cupy.any(U))
    if n_steps == 1:
        yout = cupy.squeeze(xout @ C.T)
        if not no_input:
            yout += cupy.squeeze(U @ D.T)
        return (T, cupy.squeeze(yout), cupy.squeeze(xout))
    dt = T[1] - T[0]
    if not cupy.allclose(cupy.diff(T), dt):
        raise ValueError('Time steps are not equally spaced.')
    if no_input:
        expAT_dt = expm(A.T * dt)
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ expAT_dt
        yout = cupy.squeeze(xout @ C.T)
        return (T, cupy.squeeze(yout), cupy.squeeze(xout))
    U = cupy.atleast_1d(U)
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[0] != n_steps:
        raise ValueError('U must have the same number of rows as elements in T.')
    if U.shape[1] != n_inputs:
        raise ValueError('System does not define that many inputs.')
    if not interp:
        M = cupy.vstack([cupy.hstack([A * dt, B * dt]), cupy.zeros((n_inputs, n_states + n_inputs))])
        expMT = expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ Ad + U[i - 1] @ Bd
    else:
        Mlst = [cupy.hstack([A * dt, B * dt, cupy.zeros((n_states, n_inputs))]), cupy.hstack([cupy.zeros((n_inputs, n_states + n_inputs)), cupy.identity(n_inputs)]), cupy.zeros((n_inputs, n_states + 2 * n_inputs))]
        M = cupy.vstack(Mlst)
        expMT = expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states + n_inputs:, :n_states]
        Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in range(1, n_steps):
            xout[i] = xout[i - 1] @ Ad + U[i - 1] @ Bd0 + U[i] @ Bd1
    yout = cupy.squeeze(xout @ C.T) + cupy.squeeze(U @ D.T)
    return (T, cupy.squeeze(yout), cupy.squeeze(xout))