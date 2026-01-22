from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_scaling_policy_bad_group(self):
    t = template_format.parse(inline_templates.as_heat_template_bad_group)
    stack = utils.parse_stack(t)
    up_policy = self.create_scaling_policy(t, stack, 'my-policy')
    ex = self.assertRaises(exception.ResourceFailure, up_policy.signal)
    self.assertIn('Alarm my-policy could not find scaling group', str(ex))