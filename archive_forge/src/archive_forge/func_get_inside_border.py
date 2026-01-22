import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_inside_border(self):
    """
        Retrieve the FT_StrokerBorder value corresponding to the 'inside'
        borders of a given outline.

        :return: The border index. FT_STROKER_BORDER_RIGHT for empty or invalid
                 outlines.
        """
    return FT_Outline_GetInsideBorder(byref(self._FT_Outline))