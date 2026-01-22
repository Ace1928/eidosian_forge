import functools
import inspect
import warnings
@functools.wraps(func1)
def new_func1(*args, **kwargs):
    warn_deprecation(fmt1)
    return func1(*args, **kwargs)