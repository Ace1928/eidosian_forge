import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def make_accessor(name):
    with ffi._lock:
        if name in library.__dict__ or name in FFILibrary.__dict__:
            return
        if name not in accessors:
            update_accessors()
            if name not in accessors:
                raise AttributeError(name)
        accessors[name](name)