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
def test_ignore_sys_exit(self):
    """Test the -e option to ignore sys.exit()"""
    src = 'import sys; sys.exit(1)'
    self.mktmp(src)
    with tt.AssertPrints('SystemExit'):
        _ip.run_line_magic('run', self.fname)
    with tt.AssertNotPrints('SystemExit'):
        _ip.run_line_magic('run', '-e %s' % self.fname)