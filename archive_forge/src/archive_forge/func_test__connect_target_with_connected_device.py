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
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test__connect_target_with_connected_device(self, mock_state, mock_rescan, mock_cli, mock_set_ctrls, mock_find_dev):
    """Test connect target when there's a connection to the subsystem."""
    self.conn_props.targets[0].portals[-1].controller = 'nvme0'
    mock_state.side_effect = ('connecting', None, 'live')
    dev_path = '/dev/nvme0n1'
    mock_find_dev.return_value = dev_path
    res = self.connector._connect_target(self.conn_props.targets[0])
    self.assertEqual(dev_path, res)
    self.assertEqual(3, mock_state.call_count)
    mock_state.assert_has_calls(3 * [mock.call()])
    mock_rescan.assert_called_once_with('nvme0')
    mock_set_ctrls.assert_called_once_with()
    mock_find_dev.assert_called_once_with()
    mock_cli.assert_not_called()