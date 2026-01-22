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
@mock.patch('time.sleep')
@mock.patch.object(linuxscsi.LinuxSCSI, '_multipath_resize_map')
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_scsi_wwn')
@mock.patch('os_brick.utils.get_device_size')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_device_info')
@mock.patch('os.path.realpath')
def test_extend_volume_with_mpath_pending(self, mock_realpath, mock_device_info, mock_device_size, mock_scsi_wwn, mock_find_mpath_path, mock_mpath_resize_map, mock_sleep, mock_valid_dev):
    """Test extending a volume where there is a multipath device fail."""
    mock_device_info.side_effect = [{'host': host, 'channel': '0', 'id': '0', 'lun': '1'} for host in ['0', '1']]
    mock_device_size.side_effect = [1024, 2048, 1024, 2048, 1024, 2048]
    wwn = '1234567890123456'
    mock_scsi_wwn.return_value = wwn
    mpath_path = '/dev/mapper/dm-uuid-mpath-%s' % wwn
    mock_find_mpath_path.return_value = mpath_path
    dm_path = '/dev/dm-5'
    mock_realpath.return_value = dm_path
    mock_mpath_resize_map.side_effect = (putils.ProcessExecutionError(stdout='timeout'), 'success')
    mock_valid_dev.return_value = True
    ret_size = self.linuxscsi.extend_volume(['/dev/fake1', '/dev/fake2'], use_multipath=True)
    self.assertEqual(2048, ret_size)
    expected_cmds = ['tee -a /sys/bus/scsi/drivers/sd/0:0:0:1/rescan', 'tee -a /sys/bus/scsi/drivers/sd/1:0:0:1/rescan', 'multipathd reconfigure']
    self.assertEqual(expected_cmds, self.cmds)
    mock_mpath_resize_map.assert_has_calls([mock.call(dm_path)] * 2)
    mock_realpath.assert_called_once_with(mpath_path)