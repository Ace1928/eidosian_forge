import os
import textwrap
from unittest import mock
import pycodestyle
from os_win._hacking import checks
from os_win.tests.unit import test_base
def test_ctypes_libs_not_used_directly(self):
    checker = checks.assert_ctypes_libs_not_used_directly
    errors = [(1, 0, 'O301')]
    code = 'ctypes.cdll.hbaapi'
    self._assert_has_errors(code, checker, expected_errors=errors)
    code = 'ctypes.windll.hbaapi.fake_func(fake_arg)'
    self._assert_has_errors(code, checker, expected_errors=errors)
    code = 'fake_var = ctypes.oledll.hbaapi.fake_func(fake_arg)'
    self._assert_has_errors(code, checker, expected_errors=errors)
    code = 'foo(ctypes.pydll.hbaapi.fake_func(fake_arg))'
    self._assert_has_errors(code, checker, expected_errors=errors)
    code = 'ctypes.cdll.LoadLibrary(fake_lib)'
    self._assert_has_errors(code, checker, expected_errors=errors)
    code = "ctypes.WinDLL('fake_lib_path')"
    self._assert_has_errors(code, checker, expected_errors=errors)
    code = 'ctypes.cdll.hbaapi'
    filename = os.path.join('os_win', 'utils', 'winapi', 'libs', 'hbaapi.py')
    self._assert_has_no_errors(code, checker, filename=filename)