from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def get_color_from_hex(s):
    """Transform a hex string color to a kivy
    :class:`~kivy.graphics.Color`.
    """
    if s.startswith('#'):
        return get_color_from_hex(s[1:])
    value = [int(x, 16) / 255.0 for x in split('([0-9a-f]{2})', s.lower()) if x != '']
    if len(value) == 3:
        value.append(1.0)
    return value