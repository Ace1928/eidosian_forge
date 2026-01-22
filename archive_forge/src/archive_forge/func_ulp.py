from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def ulp(val=np.float64(1.0)):
    """Return gap between `val` and nearest representable number of same type

    This is the value of a unit in the last place (ULP), and is similar in
    meaning to the MATLAB eps function.

    Parameters
    ----------
    val : scalar, optional
        scalar value of any numpy type.  Default is 1.0 (float64)

    Returns
    -------
    ulp_val : scalar
        gap between `val` and nearest representable number of same type

    Notes
    -----
    The wikipedia article on machine epsilon points out that the term *epsilon*
    can be used in the sense of a unit in the last place (ULP), or as the
    maximum relative rounding error.  The MATLAB ``eps`` function uses the ULP
    meaning, but this function is ``ulp`` rather than ``eps`` to avoid
    confusion between different meanings of *eps*.
    """
    val = np.array(val)
    if not np.isfinite(val):
        return np.nan
    if val.dtype.kind in 'iu':
        return 1
    aval = np.abs(val)
    info = type_info(val.dtype)
    fl2 = floor_log2(aval)
    if fl2 is None or fl2 < info['minexp']:
        fl2 = info['minexp']
    return 2 ** (fl2 - info['nmant'])