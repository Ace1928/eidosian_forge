from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_ensure_wt_provider_unavailable(self):
    self._tgutils._conn_wmi = None
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils._ensure_wt_provider_available)