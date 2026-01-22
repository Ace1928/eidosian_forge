import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
def check_call_wrapper(func):

    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        return check_call(func, *args, **kwargs)
    return myfunc