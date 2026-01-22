import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
from numba.core import config
def test_illegal_error_style_handling(self):
    new_env = os.environ.copy()
    new_env['NUMBA_CAPTURED_ERRORS'] = 'not_a_known_style'
    source_compiled = 'the source compiled'
    code = f"from numba import njit\n@njit\ndef foo():\n\tprint('{source_compiled}')\nfoo()"
    out, err = run_in_subprocess(dedent(code), env=new_env)
    expected = "Environment variable 'NUMBA_CAPTURED_ERRORS' is defined but its associated value 'not_a_known_style' could not be parsed."
    err_msg = err.decode('utf-8')
    self.assertIn(expected, err_msg)
    ex_expected = 'Invalid style in NUMBA_CAPTURED_ERRORS: not_a_known_style'
    self.assertIn(ex_expected, err_msg)
    self.assertIn(source_compiled, out.decode('utf-8'))