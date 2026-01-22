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
def test_cython_exec(self):
    self.break_and_run('os.path.join("foo", "bar")')
    self.assertEqual('[0]', self.eval_command('[a]'))
    return
    result = gdb.execute(textwrap.dedent('            cy exec\n            pass\n\n            "nothing"\n            end\n            '))
    result = self.tmpfile.read().rstrip()
    self.assertEqual('', result)