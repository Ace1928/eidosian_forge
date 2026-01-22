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
def getNodeHandler(self, node_class):
    try:
        return self._nodeHandlers[node_class]
    except KeyError:
        nodeType = node_class.__name__.upper()
    self._nodeHandlers[node_class] = handler = getattr(self, nodeType, self._unknown_handler)
    return handler