import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def handleNode(self, node, parent):
    if node is None:
        return
    if self.offset and getattr(node, 'lineno', None) is not None:
        node.lineno += self.offset[0]
        node.col_offset += self.offset[1]
    if self.futuresAllowed and self.nodeDepth == 0 and (not isinstance(node, ast.ImportFrom)) and (not self.isDocstring(node)):
        self.futuresAllowed = False
    self.nodeDepth += 1
    node._pyflakes_depth = self.nodeDepth
    node._pyflakes_parent = parent
    try:
        handler = self.getNodeHandler(node.__class__)
        handler(node)
    finally:
        self.nodeDepth -= 1