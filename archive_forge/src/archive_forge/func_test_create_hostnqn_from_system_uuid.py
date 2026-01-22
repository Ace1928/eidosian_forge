import builtins
import errno
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
import os_brick.privileged as privsep_brick
import os_brick.privileged.nvmeof as privsep_nvme
from os_brick.privileged import rootwrap
from os_brick.tests import base
@mock.patch('os.chmod')
@mock.patch.object(builtins, 'open', new_callable=mock.mock_open)
@mock.patch('os.makedirs')
@mock.patch.object(rootwrap, 'custom_execute')
def test_create_hostnqn_from_system_uuid(self, mock_exec, mock_mkdirs, mock_open, mock_chmod):
    system_uuid = 'ea841a98-444c-4abb-bd99-092b20518542'
    hostnqn = 'nqn.2014-08.org.nvmexpress:uuid:' + system_uuid
    res = privsep_nvme.create_hostnqn(system_uuid)
    mock_mkdirs.assert_called_once_with('/etc/nvme', mode=493, exist_ok=True)
    mock_exec.assert_not_called()
    mock_open.assert_called_once_with('/etc/nvme/hostnqn', 'w')
    mock_open().write.assert_called_once_with(hostnqn)
    mock_chmod.assert_called_once_with('/etc/nvme/hostnqn', 420)
    self.assertEqual(hostnqn, res)