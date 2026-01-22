import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
@mock.patch.object(time, 'sleep', autospec=True)
@mock.patch.object(node.NodeManager, 'get', autospec=True)
def test_wait_for_provision_state_several(self, mock_get, mock_sleep):
    mock_get.side_effect = [self._fake_node_for_wait('deploying', target='active'), self._fake_node_for_wait('deploying', target='active', error='Node locked'), self._fake_node_for_wait('deploying', target='active'), self._fake_node_for_wait('deploying', target='active'), self._fake_node_for_wait('active'), self._fake_node_for_wait('active')]
    self.mgr.wait_for_provision_state(['node1', 'node2'], 'active')
    mock_get.assert_has_calls([mock.call(self.mgr, 'node1', os_ironic_api_version=None, global_request_id=None), mock.call(self.mgr, 'node2', os_ironic_api_version=None, global_request_id=None)], any_order=True)
    self.assertEqual(6, mock_get.call_count)
    mock_sleep.assert_called_with(node._DEFAULT_POLL_INTERVAL)
    self.assertEqual(2, mock_sleep.call_count)