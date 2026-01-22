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
def handle_annotation_always_deferred(self, annotation, parent):
    fn = in_annotation(Checker.handleNode)
    self.deferFunction(lambda: fn(self, annotation, parent))