import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import golden_ratio
from scipy.fft import fft, ifft, fftfreq, fftshift
from ._warps import warp
from ._radon_transform import sart_projection_update
from .._shared.utils import convert_to_float
from warnings import warn
from functools import partial
def order_angles_golden_ratio(theta):
    """Order angles to reduce the amount of correlated information in
    subsequent projections.

    Parameters
    ----------
    theta : array of floats, shape (M,)
        Projection angles in degrees. Duplicate angles are not allowed.

    Returns
    -------
    indices_generator : generator yielding unsigned integers
        The returned generator yields indices into ``theta`` such that
        ``theta[indices]`` gives the approximate golden ratio ordering
        of the projections. In total, ``len(theta)`` indices are yielded.
        All non-negative integers < ``len(theta)`` are yielded exactly once.

    Notes
    -----
    The method used here is that of the golden ratio introduced
    by T. Kohler.

    References
    ----------
    .. [1] Kohler, T. "A projection access scheme for iterative
           reconstruction based on the golden section." Nuclear Science
           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.
    .. [2] Winkelmann, Stefanie, et al. "An optimal radial profile order
           based on the Golden Ratio for time-resolved MRI."
           Medical Imaging, IEEE Transactions on 26.1 (2007): 68-76.

    """
    interval = 180
    remaining_indices = list(np.argsort(theta))
    angle = theta[remaining_indices[0]]
    yield remaining_indices.pop(0)
    angle_increment = interval / golden_ratio ** 2
    while remaining_indices:
        remaining_angles = theta[remaining_indices]
        angle = (angle + angle_increment) % interval
        index_above = np.searchsorted(remaining_angles, angle)
        index_below = index_above - 1
        index_above %= len(remaining_indices)
        diff_below = abs(angle - remaining_angles[index_below])
        distance_below = min(diff_below % interval, diff_below % -interval)
        diff_above = abs(angle - remaining_angles[index_above])
        distance_above = min(diff_above % interval, diff_above % -interval)
        if distance_below < distance_above:
            yield remaining_indices.pop(index_below)
        else:
            yield remaining_indices.pop(index_above)