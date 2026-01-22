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
@mock.patch.object(parser.Stack, '_delete_credentials')
@mock.patch.object(stack_object.Stack, 'delete')
def test_purge_db_deletes_creds(self, mock_delete_stack, mock_creds_delete, mock_cr):
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    reason = 'stack delete complete'
    mock_creds_delete.return_value = (stack.COMPLETE, reason)
    stack.state_set(stack.DELETE, stack.COMPLETE, reason)
    stack.purge_db()
    self.assertTrue(mock_creds_delete.called)
    self.assertTrue(mock_delete_stack.called)