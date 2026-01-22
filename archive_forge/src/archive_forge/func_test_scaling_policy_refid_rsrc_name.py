from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.autoscaling import scaling_policy as aws_sp
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_scaling_policy_refid_rsrc_name(self):
    t = template_format.parse(as_template)
    stack = utils.parse_stack(t, params=as_params)
    rsrc = self.create_scaling_policy(t, stack, 'WebServerScaleUpPolicy')
    rsrc.resource_id = None
    self.assertEqual('WebServerScaleUpPolicy', rsrc.FnGetRefId())