from __future__ import annotations
import math as _math
from typing import TYPE_CHECKING, Any
from ._codata import value as _cd
import numpy as _np
def lambda2nu(lambda_: npt.ArrayLike) -> Any:
    """
    Convert wavelength to optical frequency

    Parameters
    ----------
    lambda_ : array_like
        Wavelength(s) to be converted.

    Returns
    -------
    nu : float or array of floats
        Equivalent optical frequency.

    Notes
    -----
    Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    Examples
    --------
    >>> from scipy.constants import lambda2nu, speed_of_light
    >>> import numpy as np
    >>> lambda2nu(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    """
    return c / _np.asanyarray(lambda_)