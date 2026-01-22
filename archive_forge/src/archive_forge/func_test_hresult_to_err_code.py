from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_hresult_to_err_code(self):
    fake_file_exists_hres = -2147024816
    file_exists_err_code = 80
    ret_val = _utils.hresult_to_err_code(fake_file_exists_hres)
    self.assertEqual(file_exists_err_code, ret_val)