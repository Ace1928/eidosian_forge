import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def copyBaseFunction(self, original):
    """Create a new base function based on an already-created function
        
        This is normally used to provide type-specific convenience versions of
        a definition created by the automated generator.
        """
    from OpenGL import wrapper, error
    if isinstance(original, _NullFunctionPointer):
        return self.nullFunction(original.__name__, original.DLL, resultType=original.restype, argTypes=original.argtypes, doc=original.__doc__, argNames=original.argNames, extension=original.extension, deprecated=original.deprecated, error_checker=original.error_checker)
    elif hasattr(original, 'originalFunction'):
        original = original.originalFunction
    return self.createBaseFunction(original.__name__, original.DLL, resultType=original.restype, argTypes=original.argtypes, doc=original.__doc__, argNames=original.argNames, extension=original.extension, deprecated=original.deprecated, error_checker=original.errcheck)