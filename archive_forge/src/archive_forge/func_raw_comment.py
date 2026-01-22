from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def raw_comment(self):
    """Returns the raw comment text associated with that Cursor"""
    return conf.lib.clang_Cursor_getRawCommentText(self)