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
def isDocstring(self, node):
    """
        Determine if the given node is a docstring, as long as it is at the
        correct place in the node tree.
        """
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)