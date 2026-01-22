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
def isLiteralTupleUnpacking(self, node):
    if isinstance(node, ast.Assign):
        for child in node.targets + [node.value]:
            if not hasattr(child, 'elts'):
                return False
        return True