from __future__ import (absolute_import, division, print_function)
import re
import sys
from ansible import constants as C
def stringc(text, color, wrap_nonvisible_chars=False):
    """String in color."""
    if ANSIBLE_COLOR:
        color_code = parsecolor(color)
        fmt = u'\x1b[%sm%s\x1b[0m'
        if wrap_nonvisible_chars:
            fmt = u'\x01\x1b[%sm\x02%s\x01\x1b[0m\x02'
        return u'\n'.join([fmt % (color_code, t) for t in text.split(u'\n')])
    else:
        return text