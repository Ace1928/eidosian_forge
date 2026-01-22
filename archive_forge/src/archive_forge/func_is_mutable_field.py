from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_mutable_field(self):
    """Returns True if the cursor refers to a C++ field that is declared
        'mutable'.
        """
    return conf.lib.clang_CXXField_isMutable(self)