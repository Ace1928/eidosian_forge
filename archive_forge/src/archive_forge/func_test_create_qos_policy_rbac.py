from unittest import mock
import yaml
from neutronclient.common import exceptions
from heat.common import exception
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_create_qos_policy_rbac(self):
    self._test_create(obj_type='qos_policy')