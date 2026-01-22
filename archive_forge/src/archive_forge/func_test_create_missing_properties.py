from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.resources.openstack.octavia import l7policy
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_create_missing_properties(self):
    for prop in ('action', 'listener'):
        tmpl = yaml.safe_load(inline_templates.L7POLICY_TEMPLATE)
        del tmpl['resources']['l7policy']['properties'][prop]
        self._create_stack(tmpl=yaml.dump(tmpl))
        self.assertRaises(exception.StackValidationFailed, self.l7policy.validate)