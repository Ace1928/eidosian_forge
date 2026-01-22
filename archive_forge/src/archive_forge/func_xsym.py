import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def xsym(sym):
    """get symbology for a 'character'"""
    op = _xsym[sym]
    if _use_unicode:
        return op[1]
    else:
        return op[0]