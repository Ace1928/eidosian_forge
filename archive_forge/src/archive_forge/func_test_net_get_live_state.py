import copy
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import net
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_net_get_live_state(self):
    tmpl = "\n        heat_template_version: rocky\n        resources:\n          net:\n            type: OS::Neutron::Net\n            properties:\n              value_specs:\n                'test:property': test_value\n         "
    t = template_format.parse(tmpl)
    stack = utils.parse_stack(t)
    show_net = self.patchobject(neutronclient.Client, 'show_network')
    show_net.return_value = {'network': {'status': 'ACTIVE'}}
    self.patchobject(neutronclient.Client, 'list_dhcp_agent_hosting_networks', return_value={'agents': [{'id': '1111'}]})
    self.patchobject(neutronclient.Client, 'create_network', return_value={'network': {'status': 'BUILD', 'subnets': [], 'qos_policy_id': 'some', 'name': 'name', 'admin_state_up': True, 'shared': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'mtu': 0}})
    rsrc = self.create_net(t, stack, 'net')
    network_resp = {'name': 'net1-net-wkkl2vwupdee', 'admin_state_up': True, 'tenant_id': '30f466e3d14b4251853899f9c26e2b66', 'mtu': 0, 'router:external': False, 'port_security_enabled': True, 'shared': False, 'qos_policy_id': 'some', 'id': u'5a4bb8a0-5077-4f8a-8140-5430370020e6', 'test:property': 'test_value_resp'}
    show_net.return_value = {'network': network_resp}
    reality = rsrc.get_live_state(rsrc.properties)
    expected = {'admin_state_up': True, 'qos_policy': 'some', 'value_specs': {'test:property': 'test_value_resp'}, 'port_security_enabled': True, 'dhcp_agent_ids': ['1111']}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in expected:
        if key == 'dhcp_agent_ids':
            self.assertEqual(set(expected[key]), set(reality[key]))
            continue
        self.assertEqual(expected[key], reality[key])