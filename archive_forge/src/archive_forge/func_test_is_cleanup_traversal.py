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
@mock.patch.object(resource.Resource, 'load')
@mock.patch.object(stack.Stack, 'time_remaining')
def test_is_cleanup_traversal(self, tr, mock_load, mock_cru, mock_crc, mock_pcr, mock_csc):
    tr.return_value = 317
    mock_load.return_value = (self.resource, self.stack, self.stack)
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    self.assertFalse(mock_cru.called)
    mock_crc.assert_called_once_with(self.resource, self.resource.stack.t.id, self.worker.engine_id, tr(), mock.ANY)