from __future__ import print_function, unicode_literals
import typing
import re
from functools import partial
from .lrucache import LRUCache
def imatch(pattern, name):
    """Test whether a name matches a wildcard pattern (case insensitive).

    Arguments:
        pattern (str): A wildcard pattern, e.g. ``"*.py"``.
        name (bool): A filename.

    Returns:
        bool: `True` if the filename matches the pattern.

    """
    try:
        re_pat = _PATTERN_CACHE[pattern, False]
    except KeyError:
        res = '(?ms)' + _translate(pattern, case_sensitive=False) + '\\Z'
        _PATTERN_CACHE[pattern, False] = re_pat = re.compile(res, re.IGNORECASE)
    return re_pat.match(name) is not None