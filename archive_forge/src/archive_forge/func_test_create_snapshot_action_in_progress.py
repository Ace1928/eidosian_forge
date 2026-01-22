from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.objects import snapshot as snapshot_objects
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(stack.Stack, 'load')
def test_create_snapshot_action_in_progress(self, mock_load):
    stack_name = 'stack_snapshot_action_in_progress'
    stk = self._create_stack(stack_name)
    mock_load.return_value = stk
    stk.state_set(stk.UPDATE, stk.IN_PROGRESS, 'test_override')
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.stack_snapshot, self.ctx, stk.identifier(), 'snap_none')
    self.assertEqual(exception.ActionInProgress, ex.exc_info[0])
    msg = 'Stack %(stack)s already has an action (%(action)s) in progress.' % {'stack': stack_name, 'action': stk.action}
    self.assertEqual(msg, str(ex.exc_info[1]))
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)