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
@mock.patch.object(nvmeof, 'blk_property')
@mock.patch.object(nvmeof.NVMeOFConnector, '_get_sizes_from_lba')
@mock.patch.object(nvmeof.NVMeOFConnector, '_execute')
@mock.patch.object(nvmeof.NVMeOFConnector, 'get_volume_paths')
@mock.patch('os_brick.utils.get_device_size')
def test_extend_volume_unreplicated(self, mock_device_size, mock_paths, mock_exec, mock_lba, mock_property):
    """Uses nvme to get expected size and waits until sysfs shows it."""
    new_size = 3221225472
    new_nsze = int(new_size / 512)
    old_nsze = int(new_nsze / 2)
    dev_path = '/dev/nvme0n1'
    mock_paths.return_value = [dev_path]
    stdout = '{"data": "jsondata"}'
    mock_exec.return_value = (stdout, '')
    mock_lba.return_value = (new_nsze, new_size)
    mock_property.side_effect = (str(old_nsze), str(new_nsze))
    self.assertEqual(new_size, self.connector.extend_volume(connection_properties))
    mock_paths.assert_called_with(mock.ANY)
    self.assertIsInstance(mock_paths.call_args[0][0], nvmeof.NVMeOFConnProps)
    mock_exec.assert_called_once_with('nvme', 'id-ns', '-ojson', dev_path, run_as_root=True, root_helper=self.connector._root_helper)
    mock_lba.assert_called_once_with({'data': 'jsondata'})
    self.assertEqual(2, mock_property.call_count)
    mock_property.assert_has_calls([mock.call('size', 'nvme0n1'), mock.call('size', 'nvme0n1')])
    mock_device_size.assert_not_called()