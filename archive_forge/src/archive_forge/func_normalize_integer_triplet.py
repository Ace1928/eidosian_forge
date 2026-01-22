from . import constants, types
def normalize_integer_triplet(rgb_triplet: types.IntTuple) -> types.IntegerRGB:
    """
    Normalize an integer ``rgb()`` triplet so that all values are
    within the range 0..255.

    Examples:

    .. doctest::

        >>> normalize_integer_triplet((128, 128, 128))
        IntegerRGB(red=128, green=128, blue=128)
        >>> normalize_integer_triplet((0, 0, 0))
        IntegerRGB(red=0, green=0, blue=0)
        >>> normalize_integer_triplet((255, 255, 255))
        IntegerRGB(red=255, green=255, blue=255)
        >>> normalize_integer_triplet((270, -20, -0))
        IntegerRGB(red=255, green=0, blue=0)

    :param rgb_triplet: The percentage `rgb()` triplet to normalize.

    """
    return types.IntegerRGB._make((_normalize_integer_rgb(value) for value in rgb_triplet))