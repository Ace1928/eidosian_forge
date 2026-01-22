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
@mock.patch.object(linuxscsi.LinuxSCSI, 'echo_scsi_command')
@mock.patch.object(linuxscsi.LinuxSCSI, 'flush_device_io')
@mock.patch.object(os.path, 'exists', return_value=True)
def test_remove_scsi_device_force(self, exists_mock, flush_mock, echo_mock):
    """With force we'll always call delete even if flush fails."""
    exc = exception.ExceptionChainer()
    flush_mock.side_effect = Exception()
    echo_mock.side_effect = Exception()
    device = '/dev/sdc'
    self.linuxscsi.remove_scsi_device(device, force=True, exc=exc)
    self.assertTrue(exc)
    flush_mock.assert_called_once_with(device)
    echo_mock.assert_called_once_with('/sys/block/sdc/device/delete', '1')