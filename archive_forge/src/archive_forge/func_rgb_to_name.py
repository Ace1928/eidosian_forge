from . import constants, normalization, types
def rgb_to_name(rgb_triplet: types.IntTuple, spec: str=constants.CSS3) -> str:
    """
    Convert a 3-:class:`tuple` of :class:`int`, suitable for use in an ``rgb()``
    color triplet, to its corresponding normalized color name, if any
    such name exists.

    To determine the name, the triplet will be converted to a
    normalized hexadecimal value.

    .. note:: **Spelling variants**

       Some values representing named gray colors can map to either of two names in
       CSS3, because it supports both ``"gray"`` and ``"grey"`` spelling variants for
       those colors. This function will always return the variant spelled ``"gray"``
       (such as ``"lightgray"`` instead of ``"lightgrey"``). See :ref:`the documentation
       on name conventions <color-name-conventions>` for details.

    Examples:

    .. doctest::

        >>> rgb_to_name((255, 255, 255))
        'white'
        >>> rgb_to_name((0, 0, 128))
        'navy'

    :param rgb_triplet: The ``rgb()`` triplet.
    :param spec: The specification from which to draw the list of color
       names. Default is :data:`CSS3`.
    :raises ValueError: when the given color has no name in the given spec.

    """
    return hex_to_name(rgb_to_hex(normalization.normalize_integer_triplet(rgb_triplet)), spec=spec)