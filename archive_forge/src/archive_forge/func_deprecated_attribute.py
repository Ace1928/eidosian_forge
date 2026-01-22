import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
def deprecated_attribute(name, since, message=None, alternative=None, pending=False, warning_type=DeprecationWarning):
    """
    Used to mark a public attribute as deprecated.  This creates a
    property that will warn when the given attribute name is accessed.
    To prevent the warning (i.e. for internal code), use the private
    name for the attribute by prepending an underscore
    (i.e. ``self._name``).

    Parameters
    ----------
    name : str
        The name of the deprecated attribute.

    since : str
        The release at which this API became deprecated.  This is
        required.

    message : str, optional
        Override the default deprecation message.  The format
        specifier ``name`` may be used for the name of the attribute,
        and ``alternative`` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function.

    alternative : str, optional
        An alternative attribute that the user may use in place of the
        deprecated attribute.  The deprecation warning will tell the
        user about this alternative if provided.

    pending : bool, optional
        If True, uses a AstropyPendingDeprecationWarning instead of
        ``warning_type``.

    warning_type : Warning
        Warning to be issued.
        Default is `~astropy.utils.exceptions.AstropyDeprecationWarning`.

    Examples
    --------

    ::

        class MyClass:
            # Mark the old_name as deprecated
            old_name = misc.deprecated_attribute('old_name', '0.1')

            def method(self):
                self._old_name = 42
    """
    private_name = '_' + name
    specific_deprecated = deprecated(since, name=name, obj_type='attribute', message=message, alternative=alternative, pending=pending, warning_type=warning_type)

    @specific_deprecated
    def get(self):
        return getattr(self, private_name)

    @specific_deprecated
    def set(self, val):
        setattr(self, private_name, val)

    @specific_deprecated
    def delete(self):
        delattr(self, private_name)
    return property(get, set, delete)