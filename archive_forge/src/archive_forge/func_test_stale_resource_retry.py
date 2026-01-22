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
@mock.patch.object(resource.Resource, 'state_set')
def test_stale_resource_retry(self, mock_ss, mock_cru, mock_crc, mock_pcr, mock_csc):
    current_template_id = self.resource.current_template_id
    res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, current_template_id)
    self.assertTrue(res)
    mock_ss.assert_not_called()