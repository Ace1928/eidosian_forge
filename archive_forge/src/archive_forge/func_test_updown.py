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
def test_updown(self):
    self.break_and_run('os.path.join("foo", "bar")')
    gdb.execute('cy step')
    self.assertRaises(RuntimeError, gdb.execute, 'cy down')
    result = gdb.execute('cy up', to_string=True)
    assert 'spam()' in result
    assert 'os.path.join("foo", "bar")' in result