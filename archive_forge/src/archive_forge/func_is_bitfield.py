from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_bitfield(self):
    """
        Check if the field is a bitfield.
        """
    return conf.lib.clang_Cursor_isBitField(self)