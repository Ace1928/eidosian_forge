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
def test__try_disconnect_no_disconnect(self, mock_can_disconnect, mock_execute):
    """Doesn't disconnect when it would break other devices."""
    mock_can_disconnect.return_value = False
    portal = self.conn_props.targets[0].portals[0]
    self.connector._try_disconnect(portal)
    mock_can_disconnect.assert_called_once_with()
    mock_execute.assert_not_called()