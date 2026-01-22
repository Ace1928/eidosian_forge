import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def getFIF(self, filename, mode, bb=None):
    """Get the freeimage Format (FIF) from a given filename.
        If mode is 'r', will try to determine the format by reading
        the file, otherwise only the filename is used.

        This function also tests whether the format supports reading/writing.
        """
    with self as lib:
        ftype = -1
        if mode not in 'rw':
            raise ValueError('Invalid mode (must be "r" or "w").')
        if mode == 'r':
            if bb is not None:
                fimemory = lib.FreeImage_OpenMemory(ctypes.c_char_p(bb), len(bb))
                ftype = lib.FreeImage_GetFileTypeFromMemory(ctypes.c_void_p(fimemory), len(bb))
                lib.FreeImage_CloseMemory(ctypes.c_void_p(fimemory))
            if ftype == -1 and os.path.isfile(filename):
                ftype = lib.FreeImage_GetFileType(efn(filename), 0)
        if ftype == -1:
            ftype = lib.FreeImage_GetFIFFromFilename(efn(filename))
        if ftype == -1:
            raise ValueError('Cannot determine format of file "%s"' % filename)
        elif mode == 'w' and (not lib.FreeImage_FIFSupportsWriting(ftype)):
            raise ValueError('Cannot write the format of file "%s"' % filename)
        elif mode == 'r' and (not lib.FreeImage_FIFSupportsReading(ftype)):
            raise ValueError('Cannot read the format of file "%s"' % filename)
        return ftype