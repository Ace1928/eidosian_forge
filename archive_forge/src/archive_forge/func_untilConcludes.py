from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def untilConcludes(f, *a, **kw):
    """
    Call C{f} with the given arguments, handling C{EINTR} by retrying.

    @param f: A function to call.

    @param a: Positional arguments to pass to C{f}.

    @param kw: Keyword arguments to pass to C{f}.

    @return: Whatever C{f} returns.

    @raise Exception: Whatever C{f} raises, except for C{OSError} with
        C{errno} set to C{EINTR}.
    """
    while True:
        try:
            return f(*a, **kw)
        except OSError as e:
            if e.args[0] == errno.EINTR:
                continue
            raise