import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
def interprocess_locked(path):
    """Acquires & releases a interprocess lock around call into
       decorated function."""
    lock = InterProcessLock(path)

    def decorator(f):

        @six.wraps(f)
        def wrapper(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)
        return wrapper
    return decorator