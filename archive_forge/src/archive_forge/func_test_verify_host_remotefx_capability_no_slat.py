from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_verify_host_remotefx_capability_no_slat(self):
    self._set_verify_host_remotefx_capability_mocks(isSlatCapable=False)
    self.assertRaises(exceptions.HyperVRemoteFXException, self._hostutils.verify_host_remotefx_capability)