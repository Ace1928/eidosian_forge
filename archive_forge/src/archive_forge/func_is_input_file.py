from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def is_input_file(self):
    """True if the included file is the input file."""
    return self.depth == 0