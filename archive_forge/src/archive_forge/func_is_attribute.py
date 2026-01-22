from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_attribute(self):
    """Test if this is an attribute kind."""
    return conf.lib.clang_isAttribute(self)