from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_bitfield_width(self):
    """
        Retrieve the width of a bitfield.
        """
    return conf.lib.clang_getFieldDeclBitWidth(self)