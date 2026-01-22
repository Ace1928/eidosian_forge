from __future__ import (absolute_import, division, print_function)
import re
import sys
from ansible import constants as C
def parsecolor(color):
    """SGR parameter string for the specified color name."""
    matches = re.match('color(?P<color>[0-9]+)|(?P<rgb>rgb(?P<red>[0-5])(?P<green>[0-5])(?P<blue>[0-5]))|gray(?P<gray>[0-9]+)', color)
    if not matches:
        return C.COLOR_CODES[color]
    if matches.group('color'):
        return u'38;5;%d' % int(matches.group('color'))
    if matches.group('rgb'):
        return u'38;5;%d' % (16 + 36 * int(matches.group('red')) + 6 * int(matches.group('green')) + int(matches.group('blue')))
    if matches.group('gray'):
        return u'38;5;%d' % (232 + int(matches.group('gray')))