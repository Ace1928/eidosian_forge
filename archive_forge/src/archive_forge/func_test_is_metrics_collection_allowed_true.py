from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_is_port_vm_started')
def test_is_metrics_collection_allowed_true(self, mock_is_started):
    mock_acl = mock.MagicMock()
    mock_acl.Action = self.netutils._ACL_ACTION_METER
    self._test_is_metrics_collection_allowed(mock_vm_started=mock_is_started, acls=[mock_acl, mock_acl], expected_result=True)