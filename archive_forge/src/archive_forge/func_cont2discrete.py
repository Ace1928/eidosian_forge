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
def cont2discrete(system, dt, method='zoh', alpha=None):
    """
    Transform a continuous to a discrete state-space system.

    Parameters
    ----------
    system : a tuple describing the system or an instance of `lti`
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1: (instance of `lti`)
            * 2: (num, den)
            * 3: (zeros, poles, gain)
            * 4: (A, B, C, D)

    dt : float
        The discretization time step.
    method : str, optional
        Which method to use:

            * gbt: generalized bilinear transformation
            * bilinear: Tustin's approximation ("gbt" with alpha=0.5)
            * euler: Euler (or forward differencing) method
              ("gbt" with alpha=0)
            * backward_diff: Backwards differencing ("gbt" with alpha=1.0)
            * zoh: zero-order hold (default)
            * foh: first-order hold (*versionadded: 1.3.0*)
            * impulse: equivalent impulse response (*versionadded: 1.3.0*)

    alpha : float within [0, 1], optional
        The generalized bilinear transformation weighting parameter, which
        should only be specified with method="gbt", and is ignored otherwise

    Returns
    -------
    sysd : tuple containing the discrete system
        Based on the input type, the output will be of the form

        * (num, den, dt)   for transfer function input
        * (zeros, poles, gain, dt)   for zeros-poles-gain input
        * (A, B, C, D, dt) for state-space system input

    Notes
    -----
    By default, the routine uses a Zero-Order Hold (zoh) method to perform
    the transformation. Alternatively, a generalized bilinear transformation
    may be used, which includes the common Tustin's bilinear approximation,
    an Euler's method technique, or a backwards differencing technique.

    See Also
    --------
    scipy.signal.cont2discrete


    """
    if len(system) == 1:
        return system.to_discrete()
    if len(system) == 2:
        sysd = cont2discrete(tf2ss(system[0], system[1]), dt, method=method, alpha=alpha)
        return ss2tf(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(system) == 3:
        sysd = cont2discrete(zpk2ss(system[0], system[1], system[2]), dt, method=method, alpha=alpha)
        return ss2zpk(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(system) == 4:
        a, b, c, d = system
    else:
        raise ValueError('First argument must either be a tuple of 2 (tf), 3 (zpk), or 4 (ss) arrays.')
    if method == 'gbt':
        if alpha is None:
            raise ValueError('Alpha parameter must be specified for the generalized bilinear transform (gbt) method')
        elif alpha < 0 or alpha > 1:
            raise ValueError('Alpha parameter must be within the interval [0,1] for the gbt method')
    if method == 'gbt':
        ima = cupy.eye(a.shape[0]) - alpha * dt * a
        rhs = cupy.eye(a.shape[0]) + (1.0 - alpha) * dt * a
        ad = cupy.linalg.solve(ima, rhs)
        bd = cupy.linalg.solve(ima, dt * b)
        cd = cupy.linalg.solve(ima.T, c.T)
        cd = cd.T
        dd = d + alpha * (c @ bd)
    elif method == 'bilinear' or method == 'tustin':
        return cont2discrete(system, dt, method='gbt', alpha=0.5)
    elif method == 'euler' or method == 'forward_diff':
        return cont2discrete(system, dt, method='gbt', alpha=0.0)
    elif method == 'backward_diff':
        return cont2discrete(system, dt, method='gbt', alpha=1.0)
    elif method == 'zoh':
        em_upper = cupy.hstack((a, b))
        em_lower = cupy.hstack((cupy.zeros((b.shape[1], a.shape[0])), cupy.zeros((b.shape[1], b.shape[1]))))
        em = cupy.vstack((em_upper, em_lower))
        ms = expm(dt * em)
        ms = ms[:a.shape[0], :]
        ad = ms[:, 0:a.shape[1]]
        bd = ms[:, a.shape[1]:]
        cd = c
        dd = d
    elif method == 'foh':
        n = a.shape[0]
        m = b.shape[1]
        em_upper = block_diag(cupy.hstack([a, b]) * dt, cupy.eye(m))
        em_lower = cupy.zeros((m, n + 2 * m))
        em = cupy.vstack([em_upper, em_lower])
        ms = linalg.expm(em)
        ms11 = ms[:n, 0:n]
        ms12 = ms[:n, n:n + m]
        ms13 = ms[:n, n + m:]
        ad = ms11
        bd = ms12 - ms13 + ms11 @ ms13
        cd = c
        dd = d + c @ ms13
    elif method == 'impulse':
        if not cupy.allclose(d, 0):
            raise ValueError('Impulse method is only applicableto strictly proper systems')
        ad = expm(a * dt)
        bd = ad @ b * dt
        cd = c
        dd = c @ b * dt
    else:
        raise ValueError("Unknown transformation method '%s'" % method)
    return (ad, bd, cd, dd, dt)