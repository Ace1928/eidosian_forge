import itertools
from collections import OrderedDict
import numpy as np
def has_valid_einsum_chars_only(einsum_str):
    """Check if ``einsum_str`` contains only valid characters for numpy einsum.

    Examples
    --------
    >>> has_valid_einsum_chars_only("abAZ")
    True

    >>> has_valid_einsum_chars_only("Över")
    False
    """
    return all(map(is_valid_einsum_char, einsum_str))