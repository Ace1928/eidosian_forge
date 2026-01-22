import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def end_subpath(self):
    """
        Close the current sub-path in the stroker.

        **Note**:

          You should call this function after 'begin_subpath'. If the subpath
          was not 'opened', this function 'draws' a single line segment to the
          start position when needed.
        """
    error = FT_Stroker_EndSubPath(self._FT_Stroker)
    if error:
        raise FT_Exception(error)