from .roundTools import otRound, nearestMultipleShortestRepr
import logging
def floatToFixedToStr(value, precisionBits):
    """Convert float to string with fixed-point rounding.

    This uses the shortest decimal representation (ie. the least
    number of fractional decimal digits) to represent the equivalent
    fixed-point number with ``precisionBits`` fractional binary digits.
    It uses nearestMultipleShortestRepr under the hood.

    >>> floatToFixedToStr(-0.61883544921875, precisionBits=14)
    '-0.61884'

    Args:
            value (float): The float value to convert.
            precisionBits (int): Number of precision bits, *up to a maximum of 16*.

    Returns:
            str: A string representation of the value.

    """
    scale = 1 << precisionBits
    return nearestMultipleShortestRepr(value, factor=1.0 / scale)