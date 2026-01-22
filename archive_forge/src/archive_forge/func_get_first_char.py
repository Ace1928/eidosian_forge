import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_first_char(self):
    """
        This function is used to return the first character code in the current
        charmap of a given face. It also returns the corresponding glyph index.

        :return: Glyph index of first character code. 0 if charmap is empty.

        **Note**:

          You should use this function with get_next_char to be able to parse
          all character codes available in a given charmap. The code should look
          like this:

          Note that 'agindex' is set to 0 if the charmap is empty. The result
          itself can be 0 in two cases: if the charmap is empty or if the value 0
          is the first valid character code.
        """
    agindex = FT_UInt()
    charcode = FT_Get_First_Char(self._FT_Face, byref(agindex))
    return (charcode, agindex.value)