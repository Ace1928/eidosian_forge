from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_set_new_disk_policy(self):
    self._diskutils.set_new_disk_policy(mock.sentinel.policy)
    setting_cls = self._diskutils._conn_storage.MSFT_StorageSetting
    setting_cls.Set.assert_called_once_with(NewDiskPolicy=mock.sentinel.policy)