import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def destination(value):
    """Modifier decorator to update a patch's destination.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    value : object
        Patch destination.

    Returns
    -------
    object
        The decorated object.
    """

    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.override['destination'] = value
        return wrapped
    return decorator