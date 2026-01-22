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
def test_net(self):
    t = template_format.parse(neutron_template)
    stack = utils.parse_stack(t)
    resource_type = 'networks'
    net_info_build = {'network': {'status': 'BUILD', 'subnets': [], 'name': 'name', 'admin_state_up': True, 'shared': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'mtu': 0}}
    net_info_active = copy.deepcopy(net_info_build)
    net_info_active['network'].update({'status': 'ACTIVE'})
    agent_info = {'agents': [{'admin_state_up': True, 'agent_type': 'DHCP agent', 'alive': True, 'binary': 'neutron-dhcp-agent', 'configurations': {'dhcp_driver': 'DummyDriver', 'dhcp_lease_duration': 86400, 'networks': 0, 'ports': 0, 'subnets': 0, 'use_namespaces': True}, 'created_at': '2014-03-20 05:12:34', 'description': None, 'heartbeat_timestamp': '2014-03-20 05:12:34', 'host': 'hostname', 'id': '28c25a04-3f73-45a7-a2b4-59e183943ddc', 'started_at': '2014-03-20 05:12:34', 'topic': 'dhcp_agent'}]}
    create_mock = self.patchobject(neutronclient.Client, 'create_network')
    create_mock.return_value = net_info_build
    list_dhcp_agent_mock = self.patchobject(neutronclient.Client, 'list_dhcp_agent_hosting_networks')
    list_dhcp_agent_mock.side_effect = [{'agents': []}, agent_info]
    add_dhcp_agent_mock = self.patchobject(neutronclient.Client, 'add_network_to_dhcp_agent')
    remove_dhcp_agent_mock = self.patchobject(neutronclient.Client, 'remove_network_from_dhcp_agent')
    replace_tag_mock = self.patchobject(neutronclient.Client, 'replace_tag')
    show_network_mock = self.patchobject(neutronclient.Client, 'show_network')
    show_network_mock.side_effect = [net_info_build, net_info_active, qe.NetworkNotFoundClient(status_code=404), net_info_active, net_info_active, qe.NetworkNotFoundClient(status_code=404)]
    update_net_mock = self.patchobject(neutronclient.Client, 'update_network')
    del_net_mock = self.patchobject(neutronclient.Client, 'delete_network')
    del_net_mock.side_effect = [None, qe.NetworkNotFoundClient(status_code=404)]
    self.patchobject(neutron.NeutronClientPlugin, 'get_qos_policy_id', return_value='0389f747-7785-4757-b7bb-2ab07e4b09c3')
    self.patchobject(stack['router'], 'FnGetRefId', return_value='792ff887-6c85-4a56-b518-23f24fa65581')
    rsrc = self.create_net(t, stack, 'network')
    create_mock.assert_called_with({'network': {'name': u'the_network', 'admin_state_up': True, 'tenant_id': u'c1210485b2424d48804aad5d39c61b8f', 'dns_domain': u'openstack.org.', 'shared': True, 'port_security_enabled': False, 'availability_zone_hints': ['az1'], 'mtu': 1500}})
    add_dhcp_agent_mock.assert_called_with('28c25a04-3f73-45a7-a2b4-59e183943ddc', {'network_id': u'fc68ea2c-b60b-4b4f-bd82-94ec81110766'})
    replace_tag_mock.assert_called_with(resource_type, 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'tags': ['tag1', 'tag2']})
    deps = stack.dependencies[stack['router_interface']]
    self.assertIn(stack['gateway'], deps)
    deps = stack.dependencies[stack['subnet']]
    self.assertIn(stack['gateway'], deps)
    rsrc.validate()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', ref_id)
    self.assertIsNone(rsrc.FnGetAtt('status'))
    self.assertEqual('ACTIVE', rsrc.FnGetAtt('status'))
    self.assertEqual(0, rsrc.FnGetAtt('mtu'))
    self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'Foo')
    prop_diff = {'name': 'mynet', 'dhcp_agent_ids': ['bb09cfcd-5277-473d-8336-d4ed8628ae68'], 'qos_policy': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), prop_diff)
    rsrc.handle_update(update_snippet, {}, prop_diff)
    update_net_mock.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'network': {'name': 'mynet', 'qos_policy_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}})
    add_dhcp_agent_mock.assert_called_with('bb09cfcd-5277-473d-8336-d4ed8628ae68', {'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'})
    remove_dhcp_agent_mock.assert_called_with('28c25a04-3f73-45a7-a2b4-59e183943ddc', 'fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    prop_diff['qos_policy'] = None
    rsrc.handle_update(update_snippet, {}, prop_diff)
    update_net_mock.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'network': {'name': 'mynet', 'qos_policy_id': None}})
    prop_diff['value_specs'] = {'port_security_enabled': True, 'mtu': 1500}
    rsrc.handle_update(update_snippet, {}, prop_diff)
    update_net_mock.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'network': {'name': 'mynet', 'port_security_enabled': True, 'qos_policy_id': None}})
    rsrc.handle_update(update_snippet, {}, {'name': None})
    update_net_mock.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'network': {'name': utils.PhysName(stack.name, 'test_net')}})
    rsrc.handle_update(update_snippet, {}, {'tags': []})
    replace_tag_mock.assert_called_with(resource_type, 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'tags': []})
    rsrc.handle_update(update_snippet, {}, {})
    scheduler.TaskRunner(rsrc.delete)()
    del_net_mock.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)