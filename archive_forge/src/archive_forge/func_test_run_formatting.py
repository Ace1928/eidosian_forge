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
def test_run_formatting(self):
    """ Test that %run -t -N<N> does not raise a TypeError for N > 1."""
    src = 'pass'
    self.mktmp(src)
    _ip.run_line_magic('run', '-t -N 1 %s' % self.fname)
    _ip.run_line_magic('run', '-t -N 10 %s' % self.fname)