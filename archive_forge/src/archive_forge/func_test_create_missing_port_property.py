import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import trunk
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def test_create_missing_port_property(self):
    t = template_format.parse(create_template)
    del t['resources']['trunk']['properties']['port']
    stack = utils.parse_stack(t)
    self.assertRaises(exception.StackValidationFailed, stack.validate)