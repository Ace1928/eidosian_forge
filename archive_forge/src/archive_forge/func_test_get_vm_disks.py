from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vm_disks(self, mock_get_element_associated_class):
    mock_vmsettings = self._lookup_vm()
    mock_rasds = self._create_mock_disks()
    mock_get_element_associated_class.return_value = mock_rasds
    disks, volumes = self._vmutils._get_vm_disks(mock_vmsettings)
    expected_calls = [mock.call(self._vmutils._conn, self._vmutils._STORAGE_ALLOC_SETTING_DATA_CLASS, element_instance_id=mock_vmsettings.InstanceID), mock.call(self._vmutils._conn, self._vmutils._RESOURCE_ALLOC_SETTING_DATA_CLASS, element_instance_id=mock_vmsettings.InstanceID)]
    mock_get_element_associated_class.assert_has_calls(expected_calls)
    self.assertEqual([mock_rasds[0]], disks)
    self.assertEqual([mock_rasds[1]], volumes)