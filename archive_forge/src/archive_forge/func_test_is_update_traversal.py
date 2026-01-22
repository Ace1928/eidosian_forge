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
def test_is_update_traversal(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    mock_cru.assert_called_once_with(self.resource, self.resource.stack.t.id, set(), self.worker.engine_id, mock.ANY, mock.ANY)
    self.assertFalse(mock_crc.called)
    expected_calls = []
    for req, fwd in self.stack.convergence_dependencies.leaves():
        expected_calls.append(mock.call.worker.propagate_check_resource.assert_called_once_with(self.ctx, mock.ANY, mock.ANY, self.stack.current_traversal, mock.ANY, self.graph_key, {}, self.is_update))
    mock_csc.assert_called_once_with(self.ctx, mock.ANY, self.stack.current_traversal, self.resource.id, mock.ANY, True)