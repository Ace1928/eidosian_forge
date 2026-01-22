from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def xyz_tristimulus_values(*, illuminant, observer, dtype=float):
    """Get the CIE XYZ tristimulus values.

    Given an illuminant and observer, this function returns the CIE XYZ tristimulus
    values [2]_ scaled such that :math:`Y = 1`.

    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function ``grDevices::convertColor`` [3]_.
    dtype: dtype, optional
        Output data type.

    Returns
    -------
    values : array
        Array with 3 elements :math:`X, Y, Z` containing the CIE XYZ tristimulus values
        of the given illuminant.

    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
    .. [2] https://en.wikipedia.org/wiki/CIE_1931_color_space#Meaning_of_X,_Y_and_Z
    .. [3] https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/convertColor

    Notes
    -----
    The CIE XYZ tristimulus values are calculated from :math:`x, y` [1]_, using the
    formula

    .. math:: X = x / y

    .. math:: Y = 1

    .. math:: Z = (1 - x - y) / y

    The only exception is the illuminant "D65" with aperture angle 2° for
    backward-compatibility reasons.

    Examples
    --------
    Get the CIE XYZ tristimulus values for a "D65" illuminant for a 10 degree field of
    view

    >>> xyz_tristimulus_values(illuminant="D65", observer="10")
    array([0.94809668, 1.        , 1.07305136])
    """
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(_illuminants[illuminant][observer], dtype=dtype)
    except KeyError:
        raise ValueError(f'Unknown illuminant/observer combination (`{illuminant}`, `{observer}`)')