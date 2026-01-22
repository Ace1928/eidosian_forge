from __future__ import absolute_import, unicode_literals
import re
import sys
def unescape_char(s):
    if s[0] == '\\':
        return s[1]
    else:
        return HTMLunescape(s)