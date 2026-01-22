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
def test_stack_suspend_nonexist(self):
    stack_name = 'service_suspend_nonexist_test_stack'
    t = template_format.parse(tools.wp_template)
    tmpl = templatem.Template(t)
    stk = stack.Stack(self.ctx, stack_name, tmpl)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.stack_suspend, self.ctx, stk.identifier())
    self.assertEqual(exception.EntityNotFound, ex.exc_info[0])