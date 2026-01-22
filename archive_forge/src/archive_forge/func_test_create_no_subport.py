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
def test_create_no_subport(self):
    t = template_format.parse(create_template)
    del t['resources']['trunk']['properties']['sub_ports']
    del t['resources']['subport_1']
    del t['resources']['subport_2']
    stack = utils.parse_stack(t)
    parent_port = stack['parent_port']
    self.patchobject(parent_port, 'get_reference_id', return_value='parent port id')
    self.find_resource_mock.return_value = 'parent port id'
    stk_defn.update_resource_data(stack.defn, parent_port.name, parent_port.node_data())
    self._create_trunk(stack)
    self.create_trunk_mock.assert_called_once_with({'trunk': {'description': 'trunk description', 'name': 'trunk name', 'port_id': 'parent port id'}})