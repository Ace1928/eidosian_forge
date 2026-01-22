from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def set_library_file(filename):
    """Set the exact location of libclang"""
    if Config.loaded:
        raise Exception('library file must be set before before using any other functionalities in libclang.')
    Config.library_file = fspath(filename)