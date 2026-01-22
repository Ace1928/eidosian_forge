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
def test_module_options(self):
    _ip.user_ns.pop('a', None)
    test_opts = '-x abc -m test'
    _ip.run_line_magic('run', '-m {0}.args {1}'.format(self.package, test_opts))
    assert _ip.user_ns['a'] == test_opts