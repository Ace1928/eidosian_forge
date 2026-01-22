from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def walk_preorder(self):
    """Depth-first preorder walk over the cursor and its descendants.

        Yields cursors.
        """
    yield self
    for child in self.get_children():
        for descendant in child.walk_preorder():
            yield descendant