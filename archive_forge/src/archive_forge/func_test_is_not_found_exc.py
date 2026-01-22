from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@ddt.data(_utils._WBEM_E_NOT_FOUND, mock.sentinel.wbem_error)
def test_is_not_found_exc(self, hresult):
    exc = test_base.FakeWMIExc(hresult=hresult)
    result = _utils._is_not_found_exc(exc)
    expected = hresult == _utils._WBEM_E_NOT_FOUND
    self.assertEqual(expected, result)