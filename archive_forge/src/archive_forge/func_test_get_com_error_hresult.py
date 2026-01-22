from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_get_com_error_hresult(self):
    fake_hres = -5
    expected_hres = (1 << 32) + fake_hres
    mock_excepinfo = [None] * 5 + [fake_hres]
    mock_com_err = mock.Mock(excepinfo=mock_excepinfo)
    ret_val = _utils.get_com_error_hresult(mock_com_err)
    self.assertEqual(expected_hres, ret_val)