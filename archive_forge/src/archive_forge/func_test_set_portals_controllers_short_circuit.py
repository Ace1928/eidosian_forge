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
@mock.patch('os_brick.utils.get_host_nqn', mock.Mock(return_value='nqn'))
@mock.patch.object(nvmeof, 'sysfs_property')
@mock.patch('glob.glob')
def test_set_portals_controllers_short_circuit(self, mock_glob, mock_sysfs):
    """Stops looking once we have found names for all portals."""
    self.target.portals[0].controller = 'nvme0'
    mock_glob.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0', '/sys/class/nvme-fabrics/ctl/nvme1', '/sys/class/nvme-fabrics/ctl/nvme2', '/sys/class/nvme-fabrics/ctl/nvme3']
    mock_sysfs.side_effect = [self.target.nqn, 'tcp', 'traddr=portal2,trsvcid=port2', 'nqn']
    self.target.set_portals_controllers()
    mock_glob.assert_called_once_with('/sys/class/nvme-fabrics/ctl/nvme*')
    expected_calls = [mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('transport', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('address', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('hostnqn', '/sys/class/nvme-fabrics/ctl/nvme1')]
    self.assertEqual(len(expected_calls), mock_sysfs.call_count)
    mock_sysfs.assert_has_calls(expected_calls)
    self.assertEqual('nvme0', self.target.portals[0].controller)
    self.assertEqual('nvme1', self.target.portals[1].controller)