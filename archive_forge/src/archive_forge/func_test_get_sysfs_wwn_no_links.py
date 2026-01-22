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
@mock.patch('glob.glob', return_value=[])
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwid')
def test_get_sysfs_wwn_no_links(self, get_wwid_mock, glob_mock):
    get_wwid_mock.return_value = ''
    devices = ['sdc']
    res = self.linuxscsi.get_sysfs_wwn(devices)
    self.assertEqual('', res)
    glob_mock.assert_called_once_with('/dev/disk/by-id/scsi-*')
    get_wwid_mock.assert_called_once_with(devices)