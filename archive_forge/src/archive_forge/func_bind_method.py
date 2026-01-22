import types
import sys
import numbers
import functools
import copy
import inspect
def bind_method(cls, name, func):
    """Bind a method to class, python 2 and python 3 compatible.

    Parameters
    ----------

    cls : type
        class to receive bound method
    name : basestring
        name of method on class instance
    func : function
        function to be bound as method

    Returns
    -------
    None
    """
    if not PY3:
        setattr(cls, name, types.MethodType(func, None, cls))
    else:
        setattr(cls, name, func)