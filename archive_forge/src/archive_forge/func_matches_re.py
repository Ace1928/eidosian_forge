import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
def matches_re(regex, flags=0, func=None):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string that doesn't match *regex*.

    :param regex: a regex string or precompiled pattern to match against
    :param int flags: flags that will be passed to the underlying re function
        (default 0)
    :param callable func: which underlying `re` function to call. Valid options
        are `re.fullmatch`, `re.search`, and `re.match`; the default ``None``
        means `re.fullmatch`. For performance reasons, the pattern is always
        precompiled using `re.compile`.

    .. versionadded:: 19.2.0
    .. versionchanged:: 21.3.0 *regex* can be a pre-compiled pattern.
    """
    valid_funcs = (re.fullmatch, None, re.search, re.match)
    if func not in valid_funcs:
        msg = "'func' must be one of {}.".format(', '.join(sorted((e and e.__name__ or 'None' for e in set(valid_funcs)))))
        raise ValueError(msg)
    if isinstance(regex, Pattern):
        if flags:
            msg = "'flags' can only be used with a string pattern; pass flags to re.compile() instead"
            raise TypeError(msg)
        pattern = regex
    else:
        pattern = re.compile(regex, flags)
    if func is re.match:
        match_func = pattern.match
    elif func is re.search:
        match_func = pattern.search
    else:
        match_func = pattern.fullmatch
    return _MatchesReValidator(pattern, match_func)