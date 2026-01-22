from unittest import mock
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(service.ThreadGroupManager, 'start')
@mock.patch.object(stack.Stack, 'load')
def test_stack_check(self, mock_load, mock_start):
    stack_name = 'service_check_test_stack'
    t = template_format.parse(tools.wp_template)
    stk = utils.parse_stack(t, stack_name=stack_name)
    stk.check = mock.Mock()
    self.patchobject(service, 'NotifyEvent')
    mock_load.return_value = stk
    mock_start.side_effect = self._mock_thread_start
    self.man.stack_check(self.ctx, stk.identifier())
    self.assertTrue(stk.check.called)
    stk.delete()