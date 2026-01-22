from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def is_color_transparent(c):
    """Return True if the alpha channel is 0."""
    if len(c) < 4:
        return False
    if float(c[3]) == 0.0:
        return True
    return False