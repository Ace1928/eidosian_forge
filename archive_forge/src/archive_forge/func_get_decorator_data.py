import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def get_decorator_data(obj, set_default=False):
    """Retrieve any decorator data from an object.

    Parameters
    ----------
    obj : object
        Object.
    set_default : bool
        If no data is found, a default one is set on the object and returned,
        otherwise ``None`` is returned.

    Returns
    -------
    gorilla.DecoratorData
        The decorator data or ``None``.
    """
    if inspect.isclass(obj):
        datas = getattr(obj, _DECORATOR_DATA, {})
        data = datas.setdefault(obj, None)
        if data is None and set_default:
            data = DecoratorData()
            datas[obj] = data
            setattr(obj, _DECORATOR_DATA, datas)
    else:
        data = getattr(obj, _DECORATOR_DATA, None)
        if data is None and set_default:
            data = DecoratorData()
            setattr(obj, _DECORATOR_DATA, data)
    return data