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
def setStoreValues(self, function=NULL):
    """Set the storage-of-arguments function for the whole wrapper"""
    if function is NULL or (ERROR_ON_COPY and (not STORE_POINTERS)):
        try:
            del self.storeValues
        except Exception:
            pass
    else:
        self.storeValues = function
    return self