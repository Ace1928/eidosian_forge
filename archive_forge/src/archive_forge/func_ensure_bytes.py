from __future__ import absolute_import, division, print_function, unicode_literals
import locale
import sys
def ensure_bytes(x, encoding=sys_encoding):
    if not isinstance(x, bytes):
        x = x.encode(encoding)
    return x