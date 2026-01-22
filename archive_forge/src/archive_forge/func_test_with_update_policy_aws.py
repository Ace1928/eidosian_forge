import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_with_update_policy_aws(self):
    t = template_format.parse(inline_templates.as_heat_template)
    ag = t['resources']['my-group']
    ag['update_policy'] = {'AutoScalingRollingUpdate': {'MinInstancesInService': '1', 'MaxBatchSize': '2', 'PauseTime': 'PT1S'}}
    tmpl = template_format.parse(json.dumps(t))
    stack = utils.parse_stack(tmpl)
    exc = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertIn('Unknown Property AutoScalingRollingUpdate', str(exc))