from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def get_hex_from_color(color):
    """Transform a kivy :class:`~kivy.graphics.Color` to a hex value::

        >>> get_hex_from_color((0, 1, 0))
        '#00ff00'
        >>> get_hex_from_color((.25, .77, .90, .5))
        '#3fc4e57f'

    .. versionadded:: 1.5.0
    """
    return '#' + ''.join(['{0:02x}'.format(int(x * 255)) for x in color])