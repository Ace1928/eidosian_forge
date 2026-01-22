from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_attached_disks(self):
    mock_scsi_ctrl_path = mock.MagicMock()
    expected_query = "SELECT * FROM %(class_name)s WHERE (ResourceSubType='%(res_sub_type)s' OR ResourceSubType='%(res_sub_type_virt)s' OR ResourceSubType='%(res_sub_type_dvd)s') AND Parent = '%(parent)s'" % {'class_name': self._vmutils._RESOURCE_ALLOC_SETTING_DATA_CLASS, 'res_sub_type': self._vmutils._PHYS_DISK_RES_SUB_TYPE, 'res_sub_type_virt': self._vmutils._DISK_DRIVE_RES_SUB_TYPE, 'res_sub_type_dvd': self._vmutils._DVD_DRIVE_RES_SUB_TYPE, 'parent': mock_scsi_ctrl_path.replace("'", "''")}
    expected_disks = self._vmutils._conn.query.return_value
    ret_disks = self._vmutils.get_attached_disks(mock_scsi_ctrl_path)
    self._vmutils._conn.query.assert_called_once_with(expected_query)
    self.assertEqual(expected_disks, ret_disks)