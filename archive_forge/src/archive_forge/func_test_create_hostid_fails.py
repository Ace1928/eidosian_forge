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
def test_create_hostid_fails(self, mock_mkdirs, mock_open, mock_chmod):
    mock_mkdirs.side_effect = OSError
    res = privsep_nvme.create_hostid(None)
    mock_mkdirs.assert_called_once_with('/etc/nvme', mode=493, exist_ok=True)
    mock_open.assert_not_called()
    mock_chmod.assert_not_called()
    self.assertIsNone(res)