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
def dstep(system, x0=None, t=None, n=None):
    """
    Step response of discrete-time system.

    Parameters
    ----------
    system : tuple of array_like
        A tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1: (instance of `dlti`)
            * 3: (num, den, dt)
            * 4: (zeros, poles, gain, dt)
            * 5: (A, B, C, D, dt)

    x0 : array_like, optional
        Initial state-vector.  Defaults to zero.
    t : array_like, optional
        Time points.  Computed if not given.
    n : int, optional
        The number of time points to compute (if `t` is not given).

    Returns
    -------
    tout : ndarray
        Output time points, as a 1-D array.
    yout : tuple of ndarray
        Step response of system.  Each element of the tuple represents
        the output of the system based on a step response to each input.

    See Also
    --------
    scipy.signal.dlstep
    step, dimpulse, dlsim, cont2discrete
    """
    if isinstance(system, dlti):
        system = system._as_ss()
    elif isinstance(system, lti):
        raise AttributeError('dstep can only be used with discrete-time dlti systems.')
    else:
        system = dlti(*system[:-1], dt=system[-1])._as_ss()
    if n is None:
        n = 100
    if t is None:
        t = cupy.linspace(0, n * system.dt, n, endpoint=False)
    else:
        t = cupy.asarray(t)
    yout = None
    for i in range(0, system.inputs):
        u = cupy.zeros((t.shape[0], system.inputs))
        u[:, i] = cupy.ones((t.shape[0],))
        one_output = dlsim(system, u, t=t, x0=x0)
        if yout is None:
            yout = (one_output[1],)
        else:
            yout = yout + (one_output[1],)
        tout = one_output[0]
    return (tout, yout)