from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def fixedToFloat(value, precisionBits):
    """Converts a fixed-point number to a float given the number of
    precision bits.

    Args:
            value (int): Number in fixed-point format.
            precisionBits (int): Number of precision bits.

    Returns:
            Floating point value.

    Examples::

            >>> import math
            >>> f = fixedToFloat(-10139, precisionBits=14)
            >>> math.isclose(f, -0.61883544921875)
            True
    """
    return value / (1 << precisionBits)