from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_included_file(self):
    """Returns the File that is included by the current inclusion cursor."""
    assert self.kind == CursorKind.INCLUSION_DIRECTIVE
    return conf.lib.clang_getIncludedFile(self)