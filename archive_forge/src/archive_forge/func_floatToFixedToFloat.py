from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def floatToFixedToFloat(value, precisionBits):
    """Converts a float to a fixed-point number and back again.

    By converting the float to fixed, rounding it, and converting it back
    to float again, this returns a floating point values which is exactly
    representable in fixed-point format.

    Note: this **is** equivalent to ``fixedToFloat(floatToFixed(value))``.

    Args:
            value (float): The input floating point value.
            precisionBits (int): Number of precision bits.

    Returns:
            float: The transformed and rounded value.

    Examples::
            >>> import math
            >>> f1 = -0.61884
            >>> f2 = floatToFixedToFloat(-0.61884, precisionBits=14)
            >>> f1 != f2
            True
            >>> math.isclose(f2, -0.61883544921875)
            True
    """
    scale = 1 << precisionBits
    return otRound(value * scale) / scale