from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_ref_qualifier(self):
    """
        Retrieve the ref-qualifier of the type.
        """
    return RefQualifierKind.from_id(conf.lib.clang_Type_getCXXRefQualifier(self))