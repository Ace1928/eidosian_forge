from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def result_as(func, constructor=None):
    """Wrap the result using the constructor.

    This decorator is intended for methods. The first arguments
    passed to func must be 'self'.

    Args:
    constructor: a constructor to call using the result of func(). If None,
    the type of self will be used."""

    def inner(self, *args, **kwargs):
        if constructor is None:
            wrap = type(self)
        else:
            wrap = constructor
        res = func(self, *args, **kwargs)
        return wrap(res)
    return inner