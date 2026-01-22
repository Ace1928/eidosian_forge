from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_missing_nic_data_by_name(self):
    nic_cls = self._vmutils._conn.Msvm_SyntheticEthernetPortSettingData
    nic_cls.return_value = []
    self.assertRaises(exceptions.HyperVvNicNotFound, self._vmutils._get_nic_data_by_name, mock.sentinel.name)