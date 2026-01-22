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
@mock.patch('os_brick.privileged.rootwrap.execute', return_value=('', ''))
def test_is_multipath_running(self, mock_exec):
    res = linuxscsi.LinuxSCSI.is_multipath_running(False, None, mock_exec)
    self.assertTrue(res)
    mock_exec.assert_called_once_with('multipathd', 'show', 'status', run_as_root=True, root_helper=None)