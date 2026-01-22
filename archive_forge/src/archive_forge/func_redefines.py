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
def redefines(self, other):
    """An Annotation doesn't define any name, so it cannot redefine one."""
    return False