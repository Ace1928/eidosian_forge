from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def versionToFixed(value):
    """Ensure a table version number is fixed-point.

    Args:
            value (str): a candidate table version number.

    Returns:
            int: A table version number, possibly corrected to fixed-point.
    """
    value = int(value, 0) if value.startswith('0') else float(value)
    value = ensureVersionIsLong(value)
    return value