from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@ddt.data(True, False)
def test_get_vswitch(self, enable_cache):
    self.netutils._enable_cache = enable_cache
    self.netutils._switches = {}
    self.netutils._conn.Msvm_VirtualEthernetSwitch.return_value = [self._FAKE_VSWITCH]
    vswitch = self.netutils._get_vswitch(self._FAKE_VSWITCH_NAME)
    expected_cache = {self._FAKE_VSWITCH_NAME: self._FAKE_VSWITCH} if enable_cache else {}
    self.assertEqual(expected_cache, self.netutils._switches)
    self.assertEqual(self._FAKE_VSWITCH, vswitch)