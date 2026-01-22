from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_cindex_library(self):
    try:
        library = cdll.LoadLibrary(self.get_filename())
    except OSError as e:
        msg = str(e) + '. To provide a path to libclang use Config.set_library_path() or Config.set_library_file().'
        raise LibclangError(msg)
    return library