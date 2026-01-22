import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def center_accent(string, accent):
    """
    Returns a string with accent inserted on the middle character. Useful to
    put combining accents on symbol names, including multi-character names.

    Parameters
    ==========

    string : string
        The string to place the accent in.
    accent : string
        The combining accent to insert

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combining_character
    .. [2] https://en.wikipedia.org/wiki/Combining_Diacritical_Marks

    """
    midpoint = len(string) // 2 + 1
    firstpart = string[:midpoint]
    secondpart = string[midpoint:]
    return firstpart + accent + secondpart