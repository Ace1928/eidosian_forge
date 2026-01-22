from unittest import mock
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from oslo_utils import timeutils
from heat.common import exception
from heat.common import service_utils
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_stack_delete_nonexist(self):
    stack_name = 'service_delete_nonexist_test_stack'
    stack = tools.get_stack(stack_name, self.ctx)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.delete_stack, self.ctx, stack.identifier())
    self.assertEqual(exception.EntityNotFound, ex.exc_info[0])