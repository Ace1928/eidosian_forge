from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def function_exists(self, name):
    try:
        getattr(self.lib, name)
    except AttributeError:
        return False
    return True