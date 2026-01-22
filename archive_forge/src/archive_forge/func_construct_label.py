import io
import math
import os
import typing
import weakref
def construct_label(style, prefix, pno) -> str:
    """Construct a label based on style, prefix and page number."""
    n_str = ''
    if style == 'D':
        n_str = str(pno)
    elif style == 'r':
        n_str = integerToRoman(pno).lower()
    elif style == 'R':
        n_str = integerToRoman(pno).upper()
    elif style == 'a':
        n_str = integerToLetter(pno).lower()
    elif style == 'A':
        n_str = integerToLetter(pno).upper()
    result = prefix + n_str
    return result