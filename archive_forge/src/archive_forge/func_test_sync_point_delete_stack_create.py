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
@mock.patch.object(parser.Stack, '_converge_create_or_update')
@mock.patch.object(sync_point_object.SyncPoint, 'delete_all_by_stack_and_traversal')
def test_sync_point_delete_stack_create(self, mock_syncpoint_del, mock_ccu, mock_cr):
    stack = parser.Stack(utils.dummy_context(), 'convg_updated_time_test', templatem.Template.create_empty_template(), convergence=True)
    stack.thread_group_mgr = tools.DummyThreadGroupManager()
    stack.converge_stack(template=stack.t, action=stack.CREATE)
    self.assertFalse(mock_syncpoint_del.called)
    self.assertTrue(mock_ccu.called)