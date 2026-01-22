from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def disable_option(self):
    """The command-line option that disables this diagnostic."""
    disable = _CXString()
    conf.lib.clang_getDiagnosticOption(self, byref(disable))
    return _CXString.from_result(disable)