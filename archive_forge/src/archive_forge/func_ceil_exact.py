from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def ceil_exact(val, flt_type):
    """Return nearest exact integer >= `val` in float type `flt_type`

    Parameters
    ----------
    val : int
        We have to pass val as an int rather than the floating point type
        because large integers cast as floating point may be rounded by the
        casting process.
    flt_type : numpy type
        numpy float type.

    Returns
    -------
    ceil_val : object
        value of same floating point type as `val`, that is the nearest exact
        integer in this type such that `floor_val` >= `val`.  Thus if `val` is
        exact in `flt_type`, `ceil_val` == `val`.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> ceil_exact(2, np.float32)
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> ceil_exact(2**24-1, np.float32) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> ceil_exact(2**24+1, np.float32) == 2**24+2
    True

    As for the numpy ceil function, negatives ceil towards inf

    >>> ceil_exact(-2**24-1, np.float32) == -2**24
    True
    """
    return -floor_exact(-val, flt_type)