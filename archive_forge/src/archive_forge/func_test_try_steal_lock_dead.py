from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.engine import check_resource
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import sync_point
from heat.engine import worker
from heat.rpc import api as rpc_api
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(check_resource.listener_client, 'EngineListenerClient')
@mock.patch.object(check_resource.resource_objects.Resource, 'get_obj')
@mock.patch.object(resource.Resource, 'state_set')
def test_try_steal_lock_dead(self, mock_ss, mock_get, mock_elc, mock_cru, mock_crc, mock_pcr, mock_csc):
    fake_res = mock.Mock()
    fake_res.engine_id = 'some-thing-else'
    mock_get.return_value = fake_res
    mock_elc.return_value.is_alive.return_value = False
    current_template_id = self.resource.current_template_id
    res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, current_template_id)
    self.assertTrue(res)
    mock_ss.assert_called_once_with(self.resource.action, resource.Resource.FAILED, mock.ANY)