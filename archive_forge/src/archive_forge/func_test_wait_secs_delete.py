from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
def test_wait_secs_delete(self):
    stk = utils.parse_stack(self.simple_template)
    action = status.ResourceStatus.DELETE
    self.assertEqual(0, stk['constant']._wait_secs(action))
    variable = stk['variable']._wait_secs(action)
    self.assertGreaterEqual(variable, 1.6)
    self.assertLessEqual(variable, 5.8)
    self.assertNotEqual(variable, stk['variable']._wait_secs(action))
    variable_prod = stk['variable_prod']._wait_secs(action)
    self.assertGreaterEqual(variable_prod, 2.0)
    self.assertLessEqual(variable_prod, 68.6)
    self.assertNotEqual(variable_prod, stk['variable_prod']._wait_secs(action))