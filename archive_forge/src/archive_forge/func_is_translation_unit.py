from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_translation_unit(self):
    """Test if this is a translation unit kind."""
    return conf.lib.clang_isTranslationUnit(self)