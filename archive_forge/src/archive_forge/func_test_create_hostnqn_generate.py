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
def test_create_hostnqn_generate(self, mock_exec, mock_mkdirs, mock_open, mock_chmod):
    hostnqn = mock.Mock()
    mock_exec.side_effect = [putils.ProcessExecutionError(exit_code=errno.ENOENT, stdout='totally exist sub-command', stderr=''), (hostnqn, mock.sentinel.err)]
    res = privsep_nvme.create_hostnqn()
    mock_mkdirs.assert_called_once_with('/etc/nvme', mode=493, exist_ok=True)
    self.assertEqual(2, mock_exec.call_count)
    mock_exec.assert_has_calls([mock.call('nvme', 'show-hostnqn'), mock.call('nvme', 'gen-hostnqn')])
    mock_open.assert_called_once_with('/etc/nvme/hostnqn', 'w')
    stripped_hostnqn = hostnqn.strip.return_value
    mock_open().write.assert_called_once_with(stripped_hostnqn)
    mock_chmod.assert_called_once_with('/etc/nvme/hostnqn', 420)
    self.assertEqual(stripped_hostnqn, res)