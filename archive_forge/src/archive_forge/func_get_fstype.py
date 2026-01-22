import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_fstype(self):
    """
        Return the fsType flags for a font (embedding permissions).

        The return value is a tuple containing the freetype enum name
        as a string and the actual flag as an int
        """
    flag = FT_Get_FSType_Flags(self._FT_Face)
    for k, v in FT_FSTYPE_XXX.items():
        if v == flag:
            return (k, v)