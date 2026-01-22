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
def print_on_call_decorator(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _debug(type(self).__name__, func.__name__)
        try:
            return func(self, *args, **kwargs)
        except Exception:
            _debug('An exception occurred:', traceback.format_exc())
            raise
    return wrapper