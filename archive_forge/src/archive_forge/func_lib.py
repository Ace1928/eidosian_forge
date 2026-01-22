from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@CachedProperty
def lib(self):
    lib = self.get_cindex_library()
    register_functions(lib, not Config.compatibility_check)
    Config.loaded = True
    return lib