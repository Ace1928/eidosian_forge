from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def fromDirectory(buildDir):
    """Builds a CompilationDatabase from the database found in buildDir"""
    errorCode = c_uint()
    try:
        cdb = conf.lib.clang_CompilationDatabase_fromDirectory(fspath(buildDir), byref(errorCode))
    except CompilationDatabaseError as e:
        raise CompilationDatabaseError(int(errorCode.value), 'CompilationDatabase loading failed')
    return cdb