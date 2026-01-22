from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_state_set_stack_snapshot(self, mock_ps):
    self.stack.state_set(self.stack.SNAPSHOT, self.stack.IN_PROGRESS, 'Snapshot started')
    self.assertTrue(mock_ps.called)
    mock_ps.reset_mock()
    self.stack.state_set(self.stack.SNAPSHOT, self.stack.COMPLETE, 'Snapshot complete')
    self.assertFalse(mock_ps.called)