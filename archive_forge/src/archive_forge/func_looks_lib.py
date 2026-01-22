import os
import sys
import ctypes
def looks_lib(fname):
    """Returns True if the given filename looks like a dynamic library.
    Based on extension, but cross-platform and more flexible.
    """
    fname = fname.lower()
    if sys.platform.startswith('win'):
        return fname.endswith('.dll')
    elif sys.platform.startswith('darwin'):
        return fname.endswith('.dylib')
    else:
        return fname.endswith('.so') or '.so.' in fname