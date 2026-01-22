import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_router_dependence(self):
    t = template_format.parse(neutron_subnet_and_external_gateway_template)
    stack = utils.parse_stack(t)
    deps = stack.dependencies[stack['subnet_external']]
    self.assertIn(stack['router'], deps)
    required_by = set(stack.dependencies.required_by(stack['router']))
    self.assertIn(stack['floating_ip'], required_by)