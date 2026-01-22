from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_template_argument_unsigned_value(self, num):
    """Returns the value of the indicated arg as an unsigned 64b integer."""
    return conf.lib.clang_Cursor_getTemplateArgumentUnsignedValue(self, num)