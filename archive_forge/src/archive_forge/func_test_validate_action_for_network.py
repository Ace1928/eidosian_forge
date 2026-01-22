from unittest import mock
import yaml
from neutronclient.common import exceptions
from heat.common import exception
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_validate_action_for_network(self):
    msg = 'Property error: resources.rbac.properties.action: "invalid" is not an allowed value'
    self._test_validate_invalid_action(msg)