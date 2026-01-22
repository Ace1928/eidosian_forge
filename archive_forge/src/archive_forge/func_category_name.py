from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def category_name(self):
    """The string name of the category for this diagnostic."""
    return conf.lib.clang_getDiagnosticCategoryText(self)