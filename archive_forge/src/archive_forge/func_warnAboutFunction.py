from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def warnAboutFunction(offender, warningString):
    """
    Issue a warning string, identifying C{offender} as the responsible code.

    This function is used to deprecate some behavior of a function.  It differs
    from L{warnings.warn} in that it is not limited to deprecating the behavior
    of a function currently on the call stack.

    @param offender: The function that is being deprecated.

    @param warningString: The string that should be emitted by this warning.
    @type warningString: C{str}

    @since: 11.0
    """
    offenderModule = sys.modules[offender.__module__]
    warn_explicit(warningString, category=DeprecationWarning, filename=inspect.getabsfile(offenderModule), lineno=max((lineNumber for _, lineNumber in findlinestarts(offender.__code__) if lineNumber is not None)), module=offenderModule.__name__, registry=offender.__globals__.setdefault('__warningregistry__', {}), module_globals=None)