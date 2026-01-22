from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_validate_scaling_policy_error(self):
    t = template_format.parse(as_template)
    t['resources']['my-policy']['properties']['scaling_adjustment'] = 1
    t['resources']['my-policy']['properties']['adjustment_type'] = 'change_in_capacity'
    t['resources']['my-policy']['properties']['min_adjustment_step'] = 2
    stack = utils.parse_stack(t)
    ex = self.assertRaises(exception.ResourcePropertyValueDependency, stack.validate)
    self.assertIn('min_adjustment_step property should only be specified for adjustment_type with value percent_change_in_capacity.', str(ex))