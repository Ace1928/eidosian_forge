from __future__ import division
import sys
import unicodedata
from functools import reduce
def uchar_width(c):
    """Return the rendering width of a unicode character
        """
    if unicodedata.east_asian_width(c) in 'WF':
        return 2
    elif unicodedata.combining(c):
        return 0
    else:
        return 1