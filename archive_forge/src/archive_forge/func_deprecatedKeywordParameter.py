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
def deprecatedKeywordParameter(version: Version, name: str, replacement: Optional[str]=None) -> Callable[[_Tc], _Tc]:
    """
    Return a decorator that marks a keyword parameter of a callable
    as deprecated. A warning will be emitted if a caller supplies
    a value for the parameter, whether the caller uses a keyword or
    positional syntax.

    @type version: L{incremental.Version}
    @param version: The version in which the parameter will be marked as
        having been deprecated.

    @type name: L{str}
    @param name: The name of the deprecated parameter.

    @type replacement: L{str}
    @param replacement: Optional text indicating what should be used in
        place of the deprecated parameter.

    @since: Twisted 21.2.0
    """

    def wrapper(wrappee: _Tc) -> _Tc:
        warningString = _getDeprecationWarningString(f'The {name!r} parameter to {_fullyQualifiedName(wrappee)}', version, replacement=replacement)
        doc = 'The {!r} parameter was deprecated in {}'.format(name, getVersionString(version))
        if replacement:
            doc = doc + '; ' + _getReplacementString(replacement)
        doc += '.'
        params = inspect.signature(wrappee).parameters
        if name in params and params[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            parameterIndex = list(params).index(name)

            def checkDeprecatedParameter(*args, **kwargs):
                if len(args) > parameterIndex or name in kwargs:
                    warn(warningString, DeprecationWarning, stacklevel=2)
                return wrappee(*args, **kwargs)
        else:

            def checkDeprecatedParameter(*args, **kwargs):
                if name in kwargs:
                    warn(warningString, DeprecationWarning, stacklevel=2)
                return wrappee(*args, **kwargs)
        decorated = cast(_Tc, wraps(wrappee)(checkDeprecatedParameter))
        _appendToDocstring(decorated, doc)
        return decorated
    return wrapper