from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
def test_CythonModule(self):
    """test that debug information was parsed properly into data structures"""
    self.assertEqual(self.module.name, 'codefile')
    global_vars = ('c_var', 'python_var', '__name__', '__builtins__', '__doc__', '__file__')
    assert set(global_vars).issubset(self.module.globals)