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
def calculate_cArguments(cArgs):
    for i, converter in cResolvers_mapped:
        if converter is None:
            yield cArgs[i]
        else:
            try:
                yield converter(cArgs[i])
            except Exception as err:
                err.args += (converter,)
                raise