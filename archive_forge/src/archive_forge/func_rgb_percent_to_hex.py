from . import constants, normalization, types
def rgb_percent_to_hex(rgb_percent_triplet: types.PercentTuple) -> str:
    """
    Convert a 3-:class:`tuple` of percentages, suitable for use in an ``rgb()``
    color triplet, to a normalized hexadecimal color value for that
    color.

    Examples:

    .. doctest::

        >>> rgb_percent_to_hex(("100%", "100%", "0%"))
        '#ffff00'
        >>> rgb_percent_to_hex(("0%", "0%", "50%"))
        '#000080'
        >>> rgb_percent_to_hex(("85.49%", "64.71%", "12.5%"))
        '#daa520'

    :param rgb_percent_triplet: The ``rgb()`` triplet.

    """
    return rgb_to_hex(rgb_percent_to_rgb(normalization.normalize_percent_triplet(rgb_percent_triplet)))