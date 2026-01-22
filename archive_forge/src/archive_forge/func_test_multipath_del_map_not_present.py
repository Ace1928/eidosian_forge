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
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_dm_name', return_value=None)
def test_multipath_del_map_not_present(self, name_mock):
    self.linuxscsi.multipath_del_map('dm-7')
    self.assertEqual([], self.cmds)
    name_mock.assert_called_once_with('dm-7')