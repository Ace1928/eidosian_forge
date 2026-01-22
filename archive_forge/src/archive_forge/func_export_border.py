import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def export_border(self, border, outline):
    """
        Call this function after 'get_border_counts' to export the
        corresponding border to your own 'Outline' structure.

        Note that this function appends the border points and contours to your
        outline, but does not try to resize its arrays.

        :param border:  The border index.

        :param outline: The target outline.

        **Note**:

          Always call this function after get_border_counts to get sure that
          there is enough room in your 'Outline' object to receive all new
          data.

          When an outline, or a sub-path, is 'closed', the stroker generates two
          independent 'border' outlines, named 'left' and 'right'

          When the outline, or a sub-path, is 'opened', the stroker merges the
          'border' outlines with caps. The 'left' border receives all points,
          while the 'right' border becomes empty.

          Use the function export instead if you want to retrieve all borders
          at once.
        """
    FT_Stroker_ExportBorder(self._FT_Stroker, border, byref(outline._FT_Outline))