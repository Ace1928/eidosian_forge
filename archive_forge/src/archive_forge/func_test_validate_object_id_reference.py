from unittest import mock
import yaml
from neutronclient.common import exceptions
from heat.common import exception
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_validate_object_id_reference(self):
    self._create_stack(tmpl=inline_templates.RBAC_REFERENCE_TEMPLATE)
    self.patchobject(type(self.rbac), 'is_service_available', return_value=(True, None))
    self.rbac.validate()