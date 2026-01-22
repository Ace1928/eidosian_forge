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
def test_router_get_live_state(self):
    tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          router:\n            type: OS::Neutron::Router\n            properties:\n              external_gateway_info:\n                network: public\n                enable_snat: true\n              value_specs:\n                test_value_spec: spec_value\n        '
    t = template_format.parse(tmpl)
    stack = utils.parse_stack(t)
    rsrc = stack['router']
    router_resp = {'status': 'ACTIVE', 'external_gateway_info': {'network_id': '1ede231a-0b46-40fc-ab3b-8029446d0d1b', 'enable_snat': True, 'external_fixed_ips': [{'subnet_id': '8eea1723-6de7-4255-9f8a-a0ce0db8b995', 'ip_address': '10.0.3.3'}]}, 'name': 'er-router-naqzmqnzk4ej', 'admin_state_up': True, 'tenant_id': '30f466e3d14b4251853899f9c26e2b66', 'distributed': False, 'routes': [], 'ha': False, 'id': 'b047ff06-487d-48d7-a735-a54e2fd836c2', 'test_value_spec': 'spec_value'}
    rsrc.client().show_router = mock.MagicMock(return_value={'router': router_resp})
    rsrc.client().list_l3_agent_hosting_routers = mock.MagicMock(return_value={'agents': [{'id': '1234'}, {'id': '5678'}]})
    reality = rsrc.get_live_state(rsrc.properties)
    expected = {'external_gateway_info': {'network': '1ede231a-0b46-40fc-ab3b-8029446d0d1b', 'enable_snat': True}, 'admin_state_up': True, 'value_specs': {'test_value_spec': 'spec_value'}, 'l3_agent_ids': ['1234', '5678']}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in expected:
        if key == 'external_gateway_info':
            for info in expected[key]:
                self.assertEqual(expected[key][info], reality[key][info])
        self.assertEqual(expected[key], reality[key])