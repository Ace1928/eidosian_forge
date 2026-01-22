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
def runWithWarningsSuppressed(suppressedWarnings, f, *args, **kwargs):
    """
    Run C{f(*args, **kwargs)}, but with some warnings suppressed.

    Unlike L{twisted.internet.utils.runWithWarningsSuppressed}, it has no
    special support for L{twisted.internet.defer.Deferred}.

    @param suppressedWarnings: A list of arguments to pass to
        L{warnings.filterwarnings}.  Must be a sequence of 2-tuples (args,
        kwargs).

    @param f: A callable.

    @param args: Arguments for C{f}.

    @param kwargs: Keyword arguments for C{f}

    @return: The result of C{f(*args, **kwargs)}.
    """
    with warnings.catch_warnings():
        for a, kw in suppressedWarnings:
            warnings.filterwarnings(*a, **kw)
        return f(*args, **kwargs)