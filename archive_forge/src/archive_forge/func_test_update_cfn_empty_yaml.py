from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_update_cfn_empty_yaml(self):
    t = template_format.parse('\nAWSTemplateFormatVersion: 2010-09-09\nParameters:\nResources:\nOutputs:\n')
    ut = template_format.parse('\nAWSTemplateFormatVersion: 2010-09-09\nParameters:\nResources:\n  rand:\n    Type: OS::Heat::RandomString\nOutputs:\n')
    stack = self._assert_can_create(t)
    updated = parser.Stack(self.ctx, utils.random_name(), template.Template(ut))
    stack.update(updated)
    self.assertEqual((parser.Stack.UPDATE, parser.Stack.COMPLETE), stack.state)