import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def dlclose(self, lib):
    """Close a library obtained with ffi.dlopen().  After this call,
        access to functions or variables from the library will fail
        (possibly with a segmentation fault).
        """
    type(lib).__cffi_close__(lib)