from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_field_offsetof(self):
    """Returns the offsetof the FIELD_DECL pointed by this Cursor."""
    return conf.lib.clang_Cursor_getOffsetOfField(self)