from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def nhex(i):
    """normalized hex"""
    r = hex(i)
    r = r[:2] + r[2:].lower()
    if r.endswith('l'):
        r = r[:-1]
    return r