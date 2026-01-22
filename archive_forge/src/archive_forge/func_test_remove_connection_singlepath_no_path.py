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
@mock.patch.object(linuxscsi.LinuxSCSI, '_remove_scsi_symlinks')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_del_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'is_multipath_running', return_value=True)
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_sysfs_multipath_dm')
@mock.patch.object(linuxscsi.LinuxSCSI, 'wait_for_volumes_removal')
@mock.patch.object(linuxscsi.LinuxSCSI, 'remove_scsi_device')
def test_remove_connection_singlepath_no_path(self, remove_mock, wait_mock, find_dm_mock, is_mp_running_mock, mp_del_path_mock, remove_link_mock):
    find_dm_mock.return_value = None
    devices_names = ('sda', 'sdb')
    exc = exception.ExceptionChainer()
    self.linuxscsi.remove_connection(devices_names, force=mock.sentinel.Force, exc=exc)
    find_dm_mock.assert_called_once_with(devices_names)
    is_mp_running_mock.assert_called_once()
    mp_del_path_mock.assert_has_calls([mock.call('/dev/sda'), mock.call('/dev/sdb')])
    remove_mock.assert_has_calls([mock.call('/dev/sda', mock.sentinel.Force, exc, False), mock.call('/dev/sdb', mock.sentinel.Force, exc, False)])
    wait_mock.assert_called_once_with(devices_names)
    remove_link_mock.assert_called_once_with(devices_names)