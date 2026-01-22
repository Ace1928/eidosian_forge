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
@mock.patch('os.path.exists', return_value=False)
def test_flush_device_io_non_existent(self, exists_mock):
    device = '/dev/sda'
    self.linuxscsi.flush_device_io(device)
    exists_mock.assert_called_once_with(device)