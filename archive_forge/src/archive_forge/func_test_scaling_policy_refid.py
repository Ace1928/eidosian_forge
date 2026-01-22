from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_scaling_policy_refid(self):
    t = template_format.parse(as_template)
    stack = utils.parse_stack(t)
    rsrc = stack['my-policy']
    rsrc.resource_id = 'xyz'
    self.assertEqual('xyz', rsrc.FnGetRefId())