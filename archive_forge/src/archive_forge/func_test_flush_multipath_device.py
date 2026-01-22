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
def test_flush_multipath_device(self):
    dm_map_name = '3600d0230000000000e13955cc3757800'
    with mock.patch.object(self.linuxscsi, '_execute') as exec_mock:
        self.linuxscsi.flush_multipath_device(dm_map_name)
    exec_mock.assert_called_once_with('multipath', '-f', dm_map_name, run_as_root=True, attempts=3, timeout=300, interval=10, root_helper=self.linuxscsi._root_helper)