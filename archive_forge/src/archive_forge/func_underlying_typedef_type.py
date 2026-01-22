from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def underlying_typedef_type(self):
    """Return the underlying type of a typedef declaration.

        Returns a Type for the typedef this cursor is a declaration for. If
        the current cursor is not a typedef, this raises.
        """
    if not hasattr(self, '_underlying_type'):
        assert self.kind.is_declaration()
        self._underlying_type = conf.lib.clang_getTypedefDeclUnderlyingType(self)
    return self._underlying_type