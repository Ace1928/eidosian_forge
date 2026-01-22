import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_variation_info(self):
    """
        Retrieves variation space information for the current face.
        """
    if version() < (2, 8, 1):
        raise NotImplementedError('freetype-py VF support requires FreeType 2.8.1 or later')
    p_amaster = pointer(FT_MM_Var())
    error = FT_Get_MM_Var(self._FT_Face, byref(p_amaster))
    if error:
        raise FT_Exception(error)
    vsi = VariationSpaceInfo(self, p_amaster)
    FT_Done_MM_Var_func(p_amaster)
    return vsi