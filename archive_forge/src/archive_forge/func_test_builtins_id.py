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
def test_builtins_id(self):
    """Check that %run doesn't damage __builtins__ """
    _ip = get_ipython()
    bid1 = id(_ip.user_ns['__builtins__'])
    self.run_tmpfile()
    bid2 = id(_ip.user_ns['__builtins__'])
    assert bid1 == bid2