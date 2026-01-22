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
def test_file_options(self):
    src = 'import sys\na = " ".join(sys.argv[1:])\n'
    self.mktmp(src)
    test_opts = '-x 3 --verbose'
    _ip.run_line_magic('run', '{0} {1}'.format(self.fname, test_opts))
    assert _ip.user_ns['a'] == test_opts