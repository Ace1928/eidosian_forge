from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def get_com_error_hresult_missing_excepinfo(self):
    ret_val = _utils.get_com_error_hresult(None)
    self.assertIsNone(ret_val)