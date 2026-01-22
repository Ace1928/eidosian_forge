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
@ddt.data(('live', True), ('connecting', False), (None, False))
@ddt.unpack
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test_is_live(self, state, expected, mock_state):
    """Is live only returns True if the state is 'live'."""
    mock_state.return_value = state
    self.assertIs(expected, self.portal.is_live)
    mock_state.assert_called_once_with()