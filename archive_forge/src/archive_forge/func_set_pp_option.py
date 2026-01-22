import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def set_pp_option(k, v):
    if k == 'html_mode':
        if v:
            set_html_mode(True)
        else:
            set_html_mode(False)
        return True
    if k == 'fpa_pretty':
        if v:
            set_fpa_pretty(True)
        else:
            set_fpa_pretty(False)
        return True
    val = getattr(_PP, k, None)
    if val is not None:
        _z3_assert(isinstance(v, type(val)), 'Invalid pretty print option value')
        setattr(_PP, k, v)
        return True
    val = getattr(_Formatter, k, None)
    if val is not None:
        _z3_assert(isinstance(v, type(val)), 'Invalid pretty print option value')
        setattr(_Formatter, k, v)
        return True
    return False