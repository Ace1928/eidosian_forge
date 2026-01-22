import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def parse_outline(self, outline, opened):
    """
        A convenience function used to parse a whole outline with the
        stroker. The resulting outline(s) can be retrieved later by functions
        like FT_Stroker_GetCounts and FT_Stroker_Export.

        :param outline: The source outline.

        :pram opened: A boolean. If 1, the outline is treated as an open path
                      instead of a closed one.

        **Note**:

          If 'opened' is 0 (the default), the outline is treated as a closed
          path, and the stroker generates two distinct 'border' outlines.

          If 'opened' is 1, the outline is processed as an open path, and the
          stroker generates a single 'stroke' outline.

          This function calls 'rewind' automatically.
        """
    error = FT_Stroker_ParseOutline(self._FT_Stroker, byref(outline._FT_Outline), opened)
    if error:
        raise FT_Exception(error)