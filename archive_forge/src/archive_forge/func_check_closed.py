import functools
import time
import io
import win32file
import win32pipe
import pywintypes
import win32event
import win32api
def check_closed(f):

    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError('Can not reuse socket after connection was closed.')
        return f(self, *args, **kwargs)
    return wrapped