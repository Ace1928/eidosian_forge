import ctypes, logging, os, sys
from ctypes import util
import OpenGL
def loadLibrary(dllType, name, mode=ctypes.RTLD_GLOBAL):
    """Load a given library by name with the given mode
    
    dllType -- the standard ctypes pointer to a dll type, such as
        ctypes.cdll or ctypes.windll or the underlying ctypes.CDLL or 
        ctypes.WinDLL classes.
    name -- a short module name, e.g. 'GL' or 'GLU'
    mode -- ctypes.RTLD_GLOBAL or ctypes.RTLD_LOCAL,
        controls whether the module resolves names via other
        modules already loaded into this process.  GL modules
        generally need to be loaded with GLOBAL flags
    
    returns the ctypes C-module object
    """
    if isinstance(dllType, ctypes.LibraryLoader):
        dllType = dllType._dlltype
    if sys.platform.startswith('linux'):
        return _loadLibraryPosix(dllType, name, mode)
    else:
        return _loadLibraryWindows(dllType, name, mode)