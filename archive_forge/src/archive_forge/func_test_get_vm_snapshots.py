from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(None, mock.sentinel.snap1)
def test_get_vm_snapshots(self, snap_name):
    mock_snap1 = mock.Mock(ElementName=mock.sentinel.snap1)
    mock_snap2 = mock.Mock(ElementName=mock.sentinel.snap2)
    mock_vm = self._lookup_vm()
    mock_vm.associators.return_value = [mock_snap1, mock_snap2]
    snaps = self._vmutils.get_vm_snapshots(mock.sentinel.vm_name, snap_name)
    expected_snaps = [mock_snap1.path_.return_value]
    if not snap_name:
        expected_snaps += [mock_snap2.path_.return_value]
    self.assertEqual(expected_snaps, snaps)
    mock_vm.associators.assert_called_once_with(wmi_association_class=self._vmutils._VIRTUAL_SYSTEM_SNAP_ASSOC_CLASS, wmi_result_class=self._vmutils._VIRTUAL_SYSTEM_SETTING_DATA_CLASS)