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
def check_run_submodule(self, submodule, opts=''):
    _ip.user_ns.pop('x', None)
    _ip.run_line_magic('run', '{2} -m {0}.{1}'.format(self.package, submodule, opts))
    self.assertEqual(_ip.user_ns['x'], self.value, 'Variable `x` is not loaded from module `{0}`.'.format(submodule))