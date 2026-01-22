import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch('os_brick.utils.check_valid_device')
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_scsi_wwn')
@mock.patch('os_brick.utils.get_device_size')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_device_info')
def test_extend_volume_no_mpath(self, mock_device_info, mock_device_size, mock_scsi_wwn, mock_find_mpath_path, mock_valid_dev):
    """Test extending a volume where there is no multipath device."""
    fake_device = {'host': '0', 'channel': '0', 'id': '0', 'lun': '1'}
    mock_device_info.return_value = fake_device
    first_size = 1024
    second_size = 2048
    mock_device_size.side_effect = [first_size, second_size]
    wwn = '1234567890123456'
    mock_scsi_wwn.return_value = wwn
    mock_find_mpath_path.return_value = None
    mock_valid_dev.return_value = True
    ret_size = self.linuxscsi.extend_volume(['/dev/fake'])
    self.assertEqual(second_size, ret_size)
    expected_cmds = ['tee -a /sys/bus/scsi/drivers/sd/0:0:0:1/rescan']
    self.assertEqual(expected_cmds, self.cmds)