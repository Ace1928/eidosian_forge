import os
import sys
import posixpath
import urllib.parse
def enum_types(mimedb):
    i = 0
    while True:
        try:
            ctype = _winreg.EnumKey(mimedb, i)
        except OSError:
            break
        else:
            if '\x00' not in ctype:
                yield ctype
        i += 1