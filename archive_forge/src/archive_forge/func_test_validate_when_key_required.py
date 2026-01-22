from unittest import mock
import yaml
from osc_lib import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.resources.openstack.octavia import l7rule
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
def test_validate_when_key_required(self):
    tmpl = yaml.safe_load(inline_templates.L7RULE_TEMPLATE)
    props = tmpl['resources']['l7rule']['properties']
    del props['key']
    self._create_stack(tmpl=yaml.safe_dump(tmpl))
    msg = _('Property key is missing. This property should be specified for rules of HEADER and COOKIE types.')
    with mock.patch('heat.engine.clients.os.neutron.NeutronClientPlugin.has_extension', return_value=True):
        self.assertRaisesRegex(exception.StackValidationFailed, msg, self.l7rule.validate)