import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def line_width(line):
    """Unicode combining symbols (modifiers) are not ever displayed as
    separate symbols and thus should not be counted
    """
    return len(line.translate(_remove_combining))