from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def get_random_color(alpha=1.0):
    """Returns a random color (4 tuple).

    :Parameters:
        `alpha`: float, defaults to 1.0
            If alpha == 'random', a random alpha value is generated.
    """
    from random import random
    if alpha == 'random':
        return [random(), random(), random(), random()]
    else:
        return [random(), random(), random(), alpha]