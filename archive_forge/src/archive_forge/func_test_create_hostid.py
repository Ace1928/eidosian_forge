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
def test_create_hostid(self, mock_mkdirs, mock_open, mock_chmod):
    res = privsep_nvme.create_hostid('uuid')
    mock_mkdirs.assert_called_once_with('/etc/nvme', mode=493, exist_ok=True)
    mock_open.assert_called_once_with('/etc/nvme/hostid', 'w')
    mock_open().write.assert_called_once_with('uuid\n')
    mock_chmod.assert_called_once_with('/etc/nvme/hostid', 420)
    self.assertEqual('uuid', res)