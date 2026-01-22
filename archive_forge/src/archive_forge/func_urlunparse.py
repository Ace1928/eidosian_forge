from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def urlunparse(components):
    """Put a parsed URL back together again.  This may result in a
    slightly different, but equivalent URL, if the URL that was parsed
    originally had redundant delimiters, e.g. a ? with an empty query
    (the draft states that these are equivalent)."""
    scheme, netloc, url, params, query, fragment, _coerce_result = _coerce_args(*components)
    if params:
        url = '%s;%s' % (url, params)
    return _coerce_result(urlunsplit((scheme, netloc, url, query, fragment)))