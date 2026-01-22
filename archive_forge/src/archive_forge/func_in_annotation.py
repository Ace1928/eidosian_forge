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
def in_annotation(func):

    @functools.wraps(func)
    def in_annotation_func(self, *args, **kwargs):
        with self._enter_annotation():
            return func(self, *args, **kwargs)
    return in_annotation_func