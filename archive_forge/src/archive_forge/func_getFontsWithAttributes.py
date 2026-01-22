import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def getFontsWithAttributes(self, **kwds):
    """This is a general lightweight search."""
    selected = []
    for font in self._fonts:
        OK = True
        for k, v in kwds.items():
            if getattr(font, k, None) != v:
                OK = False
        if OK:
            selected.append(font)
    return selected