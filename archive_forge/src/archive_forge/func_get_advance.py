import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_advance(self, gindex, flags):
    """
        Retrieve the advance value of a given glyph outline in an FT_Face. By
        default, the unhinted advance is returned in font units.

        :param gindex: The glyph index.

        :param flags: A set of bit flags similar to those used when calling
                      FT_Load_Glyph, used to determine what kind of advances
                      you need.

        :return: The advance value, in either font units or 16.16 format.

                 If FT_LOAD_VERTICAL_LAYOUT is set, this is the vertical
                 advance corresponding to a vertical layout. Otherwise, it is
                 the horizontal advance in a horizontal layout.
        """
    padvance = FT_Fixed(0)
    error = FT_Get_Advance(self._FT_Face, gindex, flags, byref(padvance))
    if error:
        raise FT_Exception(error)
    return padvance.value