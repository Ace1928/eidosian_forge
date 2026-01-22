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
def test_do_check_resource_marks_stack_as_failed_if_stack_timesout(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_mf = self.stack.mark_failed = mock.Mock(return_value=True)
    mock_cru.side_effect = scheduler.Timeout(None, 60)
    self.is_update = True
    self.cr._do_check_resource(self.ctx, self.stack.current_traversal, self.stack.t, {}, self.is_update, self.resource, self.stack, {})
    mock_mf.assert_called_once_with(u'Timed out')