from __future__ import division
import sys
import unicodedata
from functools import reduce
def obj2unicode(obj):
    """Return a unicode representation of a python object
    """
    if isinstance(obj, unicode_type):
        return obj
    elif isinstance(obj, bytes_type):
        try:
            return unicode_type(obj, 'utf-8')
        except UnicodeDecodeError as strerror:
            sys.stderr.write("UnicodeDecodeError exception for string '%s': %s\n" % (obj, strerror))
            return unicode_type(obj, 'utf-8', 'replace')
    else:
        return unicode_type(obj)