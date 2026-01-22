import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_kerning(self, left, right, mode=FT_KERNING_DEFAULT):
    """
        Return the kerning vector between two glyphs of a same face.

        :param left: The index of the left glyph in the kern pair.

        :param right: The index of the right glyph in the kern pair.

        :param mode: See FT_Kerning_Mode for more information. Determines the scale
                     and dimension of the returned kerning vector.

        **Note**:

          Only horizontal layouts (left-to-right & right-to-left) are supported
          by this method. Other layouts, or more sophisticated kernings, are out
          of the scope of this API function -- they can be implemented through
          format-specific interfaces.
        """
    left_glyph = self.get_char_index(left)
    right_glyph = self.get_char_index(right)
    kerning = FT_Vector(0, 0)
    error = FT_Get_Kerning(self._FT_Face, left_glyph, right_glyph, mode, byref(kerning))
    if error:
        raise FT_Exception(error)
    return kerning