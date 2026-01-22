from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def keycall(key: str, getter: Callable[[str], Any]) -> PotentialCallWrapper:
    """
    Check to see if C{key} ends with parentheses ("C{()}"); if not, wrap up the
    result of C{get} in a L{PotentialCallWrapper}.  Otherwise, call the result
    of C{get} first, before wrapping it up.

    @param key: The last dotted segment of a formatting key, as parsed by
        L{Formatter.vformat}, which may end in C{()}.

    @param getter: A function which takes a string and returns some other
        object, to be formatted and stringified for a log.

    @return: A L{PotentialCallWrapper} that will wrap up the result to allow
        for subsequent usages of parens to defer execution to log-format time.
    """
    callit = key.endswith('()')
    realKey = key[:-2] if callit else key
    value = getter(realKey)
    if callit:
        value = value()
    return PotentialCallWrapper(value)