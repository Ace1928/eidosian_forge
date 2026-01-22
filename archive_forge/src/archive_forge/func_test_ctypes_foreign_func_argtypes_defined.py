import os
import textwrap
from unittest import mock
import pycodestyle
from os_win._hacking import checks
from os_win.tests.unit import test_base
def test_ctypes_foreign_func_argtypes_defined(self):
    checker = checks.assert_ctypes_foreign_func_argtypes_defined
    errors = [(1, 0, 'O302')]
    code = 'kernel32.FakeFunc(fake_arg)'
    self._assert_has_errors(code, checker, errors)
    code = 'fake_func(kernel32.FakeFunc(fake_arg))'
    self._assert_has_errors(code, checker, errors)
    code = 'kernel32.WaitNamedPipeW(x, y)'
    self._assert_has_no_errors(code, checker)
    code = '_fake_kernel32.WaitNamedPipeW(x, y)'
    self._assert_has_no_errors(code, checker)