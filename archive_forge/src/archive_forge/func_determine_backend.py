import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def determine_backend(value, dispatch_type, *, domain, only=True, coerce=False):
    """Set the backend to the first active backend that supports ``value``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend` to ensure the same backend
    is used everywhere in a block of multimethod calls.

    Parameters
    ----------
    value
        The value being tested
    dispatch_type
        The dispatch type associated with ``value``, aka
        ":ref:`marking <MarkingGlossary>`".
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.

    See Also
    --------
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    Suppose we have two backends ``BackendA`` and ``BackendB`` each supporting
    different types, ``TypeA`` and ``TypeB``. Neither supporting the other type:

    >>> with ua.set_backend(ex.BackendA):
    ...     ex.call_multimethod(ex.TypeB(), ex.TypeB())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    Now consider a multimethod that creates a new object of ``TypeA``, or
    ``TypeB`` depending on the active backend.

    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, ex.TypeA())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    ``res`` is an object of ``TypeB`` because ``BackendB`` is set in the
    innermost with statement. So, ``call_multimethod`` fails since the types
    don't match.

    Instead, we need to first find a backend suitable for all of our objects.

    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...     x = ex.TypeA()
    ...     with ua.determine_backend(x, "mark", domain="ua_examples"):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, x)
    TypeA

    """
    dispatchables = (Dispatchable(value, dispatch_type, coerce),)
    backend = _uarray.determine_backend(domain, dispatchables, coerce)
    return set_backend(backend, coerce=coerce, only=only)