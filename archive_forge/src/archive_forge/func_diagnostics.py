from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def diagnostics(self):
    """
        Return an iterable (and indexable) object containing the diagnostics.
        """

    class DiagIterator(object):

        def __init__(self, tu):
            self.tu = tu

        def __len__(self):
            return int(conf.lib.clang_getNumDiagnostics(self.tu))

        def __getitem__(self, key):
            diag = conf.lib.clang_getDiagnostic(self.tu, key)
            if not diag:
                raise IndexError
            return Diagnostic(diag)
    return DiagIterator(self)