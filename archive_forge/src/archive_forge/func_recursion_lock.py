from pprint import pformat
from .py3compat import MutableMapping
def recursion_lock(retval, lock_name='__recursion_lock__'):

    def decorator(func):

        def wrapper(self, *args, **kw):
            if getattr(self, lock_name, False):
                return retval
            setattr(self, lock_name, True)
            try:
                return func(self, *args, **kw)
            finally:
                setattr(self, lock_name, False)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator