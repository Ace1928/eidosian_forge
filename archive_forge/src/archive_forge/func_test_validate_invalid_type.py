from unittest import mock
import yaml
from neutronclient.common import exceptions
from heat.common import exception
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_validate_invalid_type(self):
    tpl = yaml.safe_load(inline_templates.RBAC_TEMPLATE)
    tpl['resources']['rbac']['properties']['object_type'] = 'networks'
    self._create_stack(tmpl=yaml.safe_dump(tpl))
    msg = '"networks" is not an allowed value'
    self.patchobject(type(self.rbac), 'is_service_available', return_value=(True, None))
    self.assertRaisesRegex(exception.StackValidationFailed, msg, self.rbac.validate)