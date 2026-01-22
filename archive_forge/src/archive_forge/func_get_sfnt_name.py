import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_sfnt_name(self, index):
    """
        Retrieve a string of the SFNT 'name' table for a given index

        :param index: The index of the 'name' string.

        **Note**:

          The 'string' array returned in the 'aname' structure is not
          null-terminated. The application should deallocate it if it is no
          longer in use.

          Use FT_Get_Sfnt_Name_Count to get the total number of available
          'name' table entries, then do a loop until you get the right
          platform, encoding, and name ID.
        """
    name = FT_SfntName()
    error = FT_Get_Sfnt_Name(self._FT_Face, index, byref(name))
    if error:
        raise FT_Exception(error)
    return SfntName(name)