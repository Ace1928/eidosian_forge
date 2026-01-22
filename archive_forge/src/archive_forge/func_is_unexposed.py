from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_unexposed(self):
    """Test if this is an unexposed kind."""
    return conf.lib.clang_isUnexposed(self)