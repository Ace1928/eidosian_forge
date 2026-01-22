import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def is_combining(sym):
    """Check whether symbol is a unicode modifier. """
    return ord(sym) in _remove_combining