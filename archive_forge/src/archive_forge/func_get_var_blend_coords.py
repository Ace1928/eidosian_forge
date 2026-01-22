import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_var_blend_coords(self):
    """
        Get the current blend coordinates (-1.0..+1.0)
        """
    vsi = self.get_variation_info()
    num_coords = len(vsi.axes)
    ft_coords = (FT_Fixed * num_coords)()
    error = FT_Get_Var_Blend_Coordinates(self._FT_Face, num_coords, byref(ft_coords))
    if error:
        raise FT_Exception(error)
    coords = tuple([ft_coords[ai] / 65536.0 for ai in range(num_coords)])
    return coords