import _string
import re as _re
from collections import ChainMap as _ChainMap
def convert_field(self, value, conversion):
    if conversion is None:
        return value
    elif conversion == 's':
        return str(value)
    elif conversion == 'r':
        return repr(value)
    elif conversion == 'a':
        return ascii(value)
    raise ValueError('Unknown conversion specifier {0!s}'.format(conversion))