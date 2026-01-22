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
@mock.patch.object(nvmeof.NVMeOFConnector, '_execute')
@mock.patch.object(nvmeof.Portal, 'can_disconnect')
def test__try_disconnect_failure(self, mock_can_disconnect, mock_execute):
    """Confirm disconnect doesn't swallow exceptions."""
    mock_can_disconnect.return_value = True
    portal = self.conn_props.targets[0].portals[0]
    portal.controller = 'nvme0'
    mock_execute.side_effect = ValueError
    self.assertRaises(ValueError, self.connector._try_disconnect, portal)
    mock_can_disconnect.assert_called_once_with()
    mock_execute.assert_called_once_with('nvme', 'disconnect', '-d', '/dev/nvme0', root_helper=self.connector._root_helper, run_as_root=True)