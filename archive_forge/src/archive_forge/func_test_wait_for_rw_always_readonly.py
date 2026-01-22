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
@mock.patch('os_brick.utils._time_sleep')
def test_wait_for_rw_always_readonly(self, mock_sleep):
    lsblk_output = '3624a93709a738ed78583fd1200143029 (dm-2)  0\nsdb                                       0\n3624a93709a738ed78583fd120014724e (dm-1)  0\nsdc                                       0\n3624a93709a738ed78583fd120014a2bb (dm-0)  1\nsdd                                       0\n3624a93709a738ed78583fd1200143029 (dm-2)  0\nsde                                       0\n3624a93709a738ed78583fd120014724e (dm-1)  0\nsdf                                       0\n3624a93709a738ed78583fd120014a2bb (dm-0)  1\nsdg                                       0\n3624a93709a738ed78583fd1200143029 (dm-2)  0\nsdh                                       0\n3624a93709a738ed78583fd120014724e (dm-1)  0\nsdi                                       0\n3624a93709a738ed78583fd120014a2bb (dm-0)  1\nsdj                                       0\n3624a93709a738ed78583fd1200143029 (dm-2)  0\nsdk                                       0\n3624a93709a738ed78583fd120014724e (dm-1)  0\nsdl                                       0\n3624a93709a738ed78583fd120014a2bb (dm-0)  1\nsdm                                       0\nvda1                                      0\nvdb                                       0\nvdb1                                      0\nloop0                                     0'
    mock_execute = mock.Mock()
    mock_execute.return_value = (lsblk_output, None)
    self.linuxscsi._execute = mock_execute
    wwn = '3624a93709a738ed78583fd120014a2bb'
    path = '/dev/disk/by-id/dm-uuid-mpath-' + wwn
    self.assertRaises(exception.BlockDeviceReadOnly, self.linuxscsi.wait_for_rw, wwn, path)
    self.assertEqual(4, mock_sleep.call_count)