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
@mock.patch.object(linuxscsi.LinuxSCSI, '_execute')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_dm_name', return_value=None)
def test_multipath_del_map(self, name_mock, exec_mock):
    exec_mock.side_effect = [putils.ProcessExecutionError, None]
    mpath_name = '3600d0230000000000e13955cc3757800'
    name_mock.side_effect = [mpath_name, mpath_name, None]
    self.linuxscsi.multipath_del_map('dm-7')
    self.assertEqual(2, exec_mock.call_count)
    exec_mock.assert_has_calls([mock.call('multipathd', 'del', 'map', mpath_name, run_as_root=True, timeout=5, root_helper=self.linuxscsi._root_helper)] * 2)
    self.assertEqual(3, name_mock.call_count)
    name_mock.assert_has_calls([mock.call('dm-7')] * 3)