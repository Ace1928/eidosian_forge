import ctypes, logging
from OpenGL import platform, error
from OpenGL._configflags import STORE_POINTERS, ERROR_ON_COPY, SIZE_1_ARRAY_UNPACK
from OpenGL import converters
from OpenGL.converters import DefaultCConverter
from OpenGL.converters import returnCArgument,returnPyArgument
from OpenGL.latebind import LateBind
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def cArgIndex(self, argName):
    """Return the C-argument index for the given argument name"""
    argNames = self.wrappedOperation.argNames
    try:
        return asList(argNames).index(argName)
    except (ValueError, IndexError):
        raise KeyError('No argument %r in argument list %r' % (argName, argNames))