import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def load_char(self, char, flags=FT_LOAD_RENDER):
    """
        A function used to load a single glyph into the glyph slot of a face
        object, according to its character code.

        :param char: The glyph's character code, according to the current
                     charmap used in the face.

        :param flags: A flag indicating what to load for this glyph. The
                      FT_LOAD_XXX constants can be used to control the glyph
                      loading process (e.g., whether the outline should be
                      scaled, whether to load bitmaps or not, whether to hint
                      the outline, etc).

        **Note**:

          This function simply calls FT_Get_Char_Index and FT_Load_Glyph.
        """
    if isinstance(char, str) and len(char) == 1:
        char = ord(char)
    if isinstance(char, str) and len(char) != 1:
        char = ord(char.decode('utf8'))
    if isinstance(char, unicode) and len(char) == 1:
        char = ord(char)
    error = FT_Load_Char(self._FT_Face, char, flags)
    if error:
        raise FT_Exception(error)