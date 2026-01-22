from . import constants, types
def normalize_hex(hex_value: str) -> str:
    """
    Normalize a hexadecimal color value to a string consisting of the
    character `#` followed by six lowercase hexadecimal digits (what
    HTML5 terms a "valid lowercase simple color").

    If the supplied value cannot be interpreted as a hexadecimal color
    value, :exc:`ValueError` is raised. See :ref:`the conventions used
    by this module <conventions>` for information on acceptable formats
    for hexadecimal values.

    Examples:

    .. doctest::

        >>> normalize_hex("#0099cc")
        '#0099cc'
        >>> normalize_hex("#0099CC")
        '#0099cc'
        >>> normalize_hex("#09c")
        '#0099cc'
        >>> normalize_hex("#09C")
        '#0099cc'
        >>> normalize_hex("#0099gg")
        Traceback (most recent call last):
            ...
        ValueError: '#0099gg' is not a valid hexadecimal color value.
        >>> normalize_hex("0099cc")
        Traceback (most recent call last):
            ...
        ValueError: '0099cc' is not a valid hexadecimal color value.

    :param hex_value: The hexadecimal color value to normalize.
    :raises ValueError: when the input is not a valid hexadecimal color value.

    """
    match = constants.HEX_COLOR_RE.match(hex_value)
    if match is None:
        raise ValueError(f'"{hex_value}" is not a valid hexadecimal color value.')
    hex_digits = match.group(1)
    if len(hex_digits) == 3:
        hex_digits = ''.join((2 * s for s in hex_digits))
    return f'#{hex_digits.lower()}'