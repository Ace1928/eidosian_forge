from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_declaration(self):
    """
        Return the cursor for the declaration of the given type.
        """
    return conf.lib.clang_getTypeDeclaration(self)