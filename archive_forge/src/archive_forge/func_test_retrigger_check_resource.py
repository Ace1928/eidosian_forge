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
@mock.patch.object(sync_point, 'sync')
def test_retrigger_check_resource(self, mock_sync, mock_cru, mock_crc, mock_pcr, mock_csc):
    resC = self.stack['C']
    expected_predecessors = {(self.stack['A'].id, True), (self.stack['B'].id, True)}
    self.cr.retrigger_check_resource(self.ctx, resC.id, self.stack)
    mock_pcr.assert_called_once_with(self.ctx, mock.ANY, resC.id, self.stack.current_traversal, mock.ANY, (resC.id, True), None, True, None)
    call_args, call_kwargs = mock_pcr.call_args
    actual_predecessors = call_args[4]
    self.assertCountEqual(expected_predecessors, actual_predecessors)