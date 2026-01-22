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
def test_run_second(self):
    """Test that running a second file doesn't clobber the first, gh-3547"""
    self.mktmp('avar = 1\ndef afunc():\n  return avar\n')
    with tt.TempFileMixin() as empty:
        empty.mktmp('')
        _ip.run_line_magic('run', self.fname)
        _ip.run_line_magic('run', empty.fname)
        assert _ip.user_ns['afunc']() == 1