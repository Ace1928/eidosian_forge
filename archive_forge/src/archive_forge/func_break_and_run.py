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
def break_and_run(self, source_line):
    break_lineno = test_libcython.source_to_lineno[source_line]
    gdb.execute('cy break codefile:%d' % break_lineno, to_string=True)
    gdb.execute('run', to_string=True)