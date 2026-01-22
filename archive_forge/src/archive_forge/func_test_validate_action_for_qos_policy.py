from unittest import mock
import yaml
from neutronclient.common import exceptions
from heat.common import exception
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_validate_action_for_qos_policy(self):
    msg = 'Property error: resources.rbac.properties.action: "invalid" is not an allowed value'
    self._test_validate_invalid_action(msg, obj_type='qos_policy')
    msg = 'Property error: resources.rbac.properties.action: Invalid action "access_as_external" for object type qos_policy. Valid actions: access_as_shared'
    self._test_validate_invalid_action(msg, obj_type='qos_policy', invalid_action='access_as_external')