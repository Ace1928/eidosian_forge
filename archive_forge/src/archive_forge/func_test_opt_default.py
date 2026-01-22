import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
@TestCase.run_test_in_subprocess()
def test_opt_default(self):
    expected = {'loop_vectorize': False, 'slp_vectorize': False, 'opt': 0, 'cost': 'cheap'}
    self.check(expected, 3, 3)