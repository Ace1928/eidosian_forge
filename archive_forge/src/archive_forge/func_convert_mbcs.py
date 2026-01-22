import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def convert_mbcs(s):
    dec = getattr(s, 'decode', None)
    if dec is not None:
        try:
            s = dec('mbcs')
        except UnicodeError:
            pass
    return s