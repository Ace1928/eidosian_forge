import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
def split_multichar(ss, chars):
    """
    Split all the strings in ss at any of the characters in chars.
    Example:

        >>> ss = ["a.string[0].with_separators"]
        >>> chars = list(".[]_")
        >>> split_multichar(ss, chars)
        ['a', 'string', '0', '', 'with', 'separators']

    :param (list) ss: A list of strings.
    :param (list) chars: Is a list of chars (note: not a string).
    """
    if len(chars) == 0:
        return ss
    c = chars.pop()
    ss = reduce(lambda x, y: x + y, map(lambda x: x.split(c), ss))
    return split_multichar(ss, chars)