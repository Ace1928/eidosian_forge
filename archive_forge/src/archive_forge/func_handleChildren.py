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
def handleChildren(self, tree, omit=None):
    for node in iter_child_nodes(tree, omit=omit):
        self.handleNode(node, tree)