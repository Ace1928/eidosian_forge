from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_align(self):
    """
        Retrieve the alignment of the record.
        """
    return conf.lib.clang_Type_getAlignOf(self)