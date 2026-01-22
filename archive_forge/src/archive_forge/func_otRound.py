import math
import functools
import logging
def otRound(value):
    """Round float value to nearest integer towards ``+Infinity``.

    The OpenType spec (in the section on `"normalization" of OpenType Font Variations <https://docs.microsoft.com/en-us/typography/opentype/spec/otvaroverview#coordinate-scales-and-normalization>`_)
    defines the required method for converting floating point values to
    fixed-point. In particular it specifies the following rounding strategy:

            for fractional values of 0.5 and higher, take the next higher integer;
            for other fractional values, truncate.

    This function rounds the floating-point value according to this strategy
    in preparation for conversion to fixed-point.

    Args:
            value (float): The input floating-point value.

    Returns
            float: The rounded value.
    """
    return int(math.floor(value + 0.5))