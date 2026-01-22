import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import openstacksdk
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
def test_deprecated_network_id(self):
    template = '\n        heat_template_version: 2015-04-30\n        resources:\n          net:\n            type: OS::Neutron::Net\n            properties:\n              name: test\n          subnet:\n            type: OS::Neutron::Subnet\n            properties:\n              network_id: { get_resource: net }\n              cidr: 10.0.0.0/24\n        '
    t = template_format.parse(template)
    stack = utils.parse_stack(t)
    rsrc = stack['subnet']
    nd = {'reference_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}
    stk_defn.update_resource_data(stack.defn, 'net', node_data.NodeData.from_dict(nd))
    self.create_mock.return_value = {'subnet': {'id': '91e47a57-7508-46fe-afc9-fc454e8580e1', 'ip_version': 4, 'name': 'name', 'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f'}}
    stack.create()
    self.assertEqual(hot_funcs.GetResource(stack.defn, 'get_resource', 'net'), rsrc.properties.get('network'))
    self.assertIsNone(rsrc.properties.get('network_id'))