from . import constants, normalization, types
def name_to_rgb_percent(name: str, spec: str=constants.CSS3) -> types.PercentRGB:
    """
    Convert a color name to a 3-:class:`tuple` of percentages suitable for use
    in an ``rgb()`` triplet specifying that color.

    The color name will be normalized to lower-case before being looked
    up.

    Examples:

    .. doctest::

        >>> name_to_rgb_percent("white")
        PercentRGB(red='100%', green='100%', blue='100%')
        >>> name_to_rgb_percent("navy")
        PercentRGB(red='0%', green='0%', blue='50%')
        >>> name_to_rgb_percent("goldenrod")
        PercentRGB(red='85.49%', green='64.71%', blue='12.5%')

    :param name: The color name to convert.
    :param spec: The specification from which to draw the list of color
       names. Default is :data:`CSS3`.
    :raises ValueError: when the given name has no definition in the given spec.

    """
    return rgb_to_rgb_percent(name_to_rgb(name, spec=spec))