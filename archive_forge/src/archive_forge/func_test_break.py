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
def test_break(self):
    breakpoint_amount = len(gdb.breakpoints() or ())
    gdb.execute('cy break codefile.spam')
    self.assertEqual(len(gdb.breakpoints()), breakpoint_amount + 1)
    bp = gdb.breakpoints()[-1]
    self.assertEqual(bp.type, gdb.BP_BREAKPOINT)
    assert self.spam_func.cname in bp.location
    assert bp.enabled