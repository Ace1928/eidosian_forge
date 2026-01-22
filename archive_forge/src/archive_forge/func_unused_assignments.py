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
def unused_assignments(self):
    """
        Return a generator for the assignments which have not been used.
        """
    for name, binding in self.items():
        if not binding.used and name != '_' and (name not in self.globals) and (not self.usesLocals) and isinstance(binding, Assignment):
            yield (name, binding)