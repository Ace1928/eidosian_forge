import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(os.path, 'exists')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_mdadm')
def test_create_raid_cmd_simple(self, mock_run_mdadm, mock_os):
    mock_os.return_value = True
    self.assertIsNone(self.connector.create_raid(['/dev/sda'], '1', 'md1', 'name', True))
    mock_run_mdadm.assert_called_with(['mdadm', '-C', '-o', 'md1', '-R', '-N', 'name', '--level', '1', '--raid-devices=1', '--bitmap=internal', '--homehost=any', '--failfast', '--assume-clean', '/dev/sda'])
    mock_os.assert_called_with('/dev/md/name')