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
@mock.patch('builtins.open')
def test_get_sysfs_wwn_mpath(self, open_mock):
    wwn = '3600d0230000000000e13955cc3757800'
    cm_open = open_mock.return_value.__enter__.return_value
    cm_open.read.return_value = 'mpath-' + wwn
    res = self.linuxscsi.get_sysfs_wwn(mock.sentinel.device_names, 'dm-1')
    open_mock.assert_called_once_with('/sys/block/dm-1/dm/uuid')
    self.assertEqual(wwn, res)