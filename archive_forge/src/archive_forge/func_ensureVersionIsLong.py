from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def ensureVersionIsLong(value):
    """Ensure a table version is an unsigned long.

    OpenType table version numbers are expressed as a single unsigned long
    comprising of an unsigned short major version and unsigned short minor
    version. This function detects if the value to be used as a version number
    looks too small (i.e. is less than ``0x10000``), and converts it to
    fixed-point using :func:`floatToFixed` if so.

    Args:
            value (Number): a candidate table version number.

    Returns:
            int: A table version number, possibly corrected to fixed-point.
    """
    if value < 65536:
        newValue = floatToFixed(value, 16)
        log.warning('Table version value is a float: %.4f; fix to use hex instead: 0x%08x', value, newValue)
        value = newValue
    return value