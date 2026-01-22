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
@mock.patch('glob.glob')
def test_get_hctl_no_target(self, glob_mock):
    glob_mock.side_effect = [[], ['/sys/class/iscsi_host/host3/device/session1', '/sys/class/iscsi_host/host3/device/session1']]
    res = self.linuxscsi.get_hctl('1', '2')
    self.assertEqual(('3', '-', '-', '2'), res)
    glob_mock.assert_has_calls([mock.call('/sys/class/iscsi_host/host*/device/session1/target*'), mock.call('/sys/class/iscsi_host/host*/device/session1')])