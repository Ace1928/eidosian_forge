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
@mock.patch.object(nvmeof.NVMeOFConnector, '_do_multipath', mock.Mock(return_value=True))
@mock.patch.object(nvmeof.Target, 'find_device')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_nvme_cli')
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test__connect_target_multipath(self, mock_state, mock_rescan, mock_cli, mock_set_ctrls, mock_find_dev):
    """Test connect when we do a new connection and find the device."""
    target = self.conn_props.targets[0]
    mock_state.side_effect = [None, None, None]
    dev_path = '/dev/nvme0n1'
    mock_find_dev.return_value = dev_path
    res = self.connector._connect_target(target)
    self.assertEqual(dev_path, res)
    self.assertEqual(3, mock_state.call_count)
    mock_state.assert_has_calls(3 * [mock.call()])
    mock_rescan.assert_not_called()
    mock_set_ctrls.assert_called_once_with()
    mock_find_dev.assert_called_once_with()
    self.assertEqual(len(target.portals), mock_cli.call_count)
    mock_cli.assert_has_calls([mock.call(['connect', '-a', portal.address, '-s', portal.port, '-t', portal.transport, '-n', target.nqn, '-Q', '128', '-l', '-1']) for portal in target.portals])