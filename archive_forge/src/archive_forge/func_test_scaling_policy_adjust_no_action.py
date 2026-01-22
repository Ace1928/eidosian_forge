from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_scaling_policy_adjust_no_action(self):
    t = template_format.parse(as_template)
    stack = utils.parse_stack(t, params=as_params)
    up_policy = self.create_scaling_policy(t, stack, 'my-policy')
    group = stack['my-group']
    self.patchobject(group, 'adjust', side_effect=resource.NoActionRequired())
    self.assertRaises(resource.NoActionRequired, up_policy.handle_signal)