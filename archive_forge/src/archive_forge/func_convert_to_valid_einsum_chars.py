import itertools
from collections import OrderedDict
import numpy as np
def convert_to_valid_einsum_chars(einsum_str):
    """Convert the str ``einsum_str`` to contain only the alphabetic characters
    valid for numpy einsum. If there are too many symbols, let the backend
    throw an error.

    Examples
    --------
    >>> oe.parser.convert_to_valid_einsum_chars("Ĥěļļö")
    'cbdda'
    """
    symbols = sorted(set(einsum_str) - set(',->'))
    replacer = {x: get_symbol(i) for i, x in enumerate(symbols)}
    return ''.join((replacer.get(x, x) for x in einsum_str))