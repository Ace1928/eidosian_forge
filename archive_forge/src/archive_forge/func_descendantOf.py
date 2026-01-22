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
def descendantOf(self, node, ancestors, stop):
    for a in ancestors:
        if self.getCommonAncestor(node, a, stop):
            return True
    return False