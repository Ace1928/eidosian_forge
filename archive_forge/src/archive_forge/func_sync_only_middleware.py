from functools import partial, update_wrapper, wraps
from asgiref.sync import iscoroutinefunction
def sync_only_middleware(func):
    """
    Mark a middleware factory as returning a sync middleware.
    This is the default.
    """
    func.sync_capable = True
    func.async_capable = False
    return func