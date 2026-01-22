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
@mock.patch('os_brick.utils.check_valid_device')
def test_extend_volume_with_mpath_path_down(self, mock_valid_dev):
    """Test extending a volume where there is a path down."""
    mock_valid_dev.return_value = False
    dev1 = '/dev/fake1'
    dev2 = '/dev/fake2'
    self.assertRaises(exception.BrickException, self.linuxscsi.extend_volume, [dev1, dev2], use_multipath=True)
    mock_valid_dev.assert_called_once_with(self.linuxscsi, dev1)