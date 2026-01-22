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
def deprecatedProperty(version, replacement=None):
    """
    Return a decorator that marks a property as deprecated. To deprecate a
    regular callable or class, see L{deprecated}.

    @type version: L{incremental.Version}
    @param version: The version in which the callable will be marked as
        having been deprecated.  The decorated function will be annotated
        with this version, having it set as its C{deprecatedVersion}
        attribute.

    @param replacement: what should be used in place of the callable.
        Either pass in a string, which will be inserted into the warning
        message, or a callable, which will be expanded to its full import
        path.
    @type replacement: C{str} or callable

    @return: A new property with deprecated setter and getter.
    @rtype: C{property}

    @since: 16.1.0
    """

    class _DeprecatedProperty(property):
        """
        Extension of the build-in property to allow deprecated setters.
        """

        def _deprecatedWrapper(self, function):

            @wraps(function)
            def deprecatedFunction(*args, **kwargs):
                warn(self.warningString, DeprecationWarning, stacklevel=2)
                return function(*args, **kwargs)
            return deprecatedFunction

        def setter(self, function):
            return property.setter(self, self._deprecatedWrapper(function))

    def deprecationDecorator(function):
        warningString = getDeprecationWarningString(function, version, None, replacement)

        @wraps(function)
        def deprecatedFunction(*args, **kwargs):
            warn(warningString, DeprecationWarning, stacklevel=2)
            return function(*args, **kwargs)
        _appendToDocstring(deprecatedFunction, _getDeprecationDocstring(version, replacement))
        deprecatedFunction.deprecatedVersion = version
        result = _DeprecatedProperty(deprecatedFunction)
        result.warningString = warningString
        return result
    return deprecationDecorator