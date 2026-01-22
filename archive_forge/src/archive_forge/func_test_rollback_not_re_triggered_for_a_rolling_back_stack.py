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
@mock.patch.object(stack.Stack, 'rollback')
def test_rollback_not_re_triggered_for_a_rolling_back_stack(self, mock_tr, mock_cru, mock_crc, mock_pcr, mock_csc):
    self.stack.disable_rollback = False
    self.stack.action = self.stack.ROLLBACK
    self.stack.status = self.stack.IN_PROGRESS
    self.stack.store()
    dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
    mock_cru.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.stack.CREATE)
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    self.assertFalse(mock_tr.called)