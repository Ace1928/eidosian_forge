import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
def test_run_i_after_reset(self):
    """Check that %run -i still works after %reset (gh-693)"""
    src = 'yy = zz\n'
    self.mktmp(src)
    _ip.run_cell('zz = 23')
    try:
        _ip.run_line_magic('run', '-i %s' % self.fname)
        assert _ip.user_ns['yy'] == 23
    finally:
        _ip.run_line_magic('reset', '-f')
    _ip.run_cell('zz = 23')
    try:
        _ip.run_line_magic('run', '-i %s' % self.fname)
        assert _ip.user_ns['yy'] == 23
    finally:
        _ip.run_line_magic('reset', '-f')