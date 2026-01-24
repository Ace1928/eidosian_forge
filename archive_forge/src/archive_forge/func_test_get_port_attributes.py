from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_get_port_attributes(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties'].pop('fixed_ips')
    stack = utils.parse_stack(t)
    subnet_dict = {'name': 'test-subnet', 'enable_dhcp': True, 'network_id': 'net1234', 'dns_nameservers': [], 'tenant_id': '58a61fc3992944ce971404a2ece6ff98', 'ipv6_ra_mode': None, 'cidr': '10.0.0.0/24', 'allocation_pools': [{'start': '10.0.0.2', 'end': u'10.0.0.254'}], 'gateway_ip': '10.0.0.1', 'ipv6_address_mode': None, 'ip_version': 4, 'host_routes': [], 'id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e'}
    network_dict = {'name': 'test-network', 'status': 'ACTIVE', 'router:external': False, 'availability_zone_hints': [], 'availability_zones': ['nova'], 'ipv4_address_scope': None, 'description': '', 'subnets': [subnet_dict['id']], 'port_security_enabled': True, 'propagate_uplink_status': True, 'tenant_id': '58a61fc3992944ce971404a2ece6ff98', 'tags': [], 'ipv6_address_scope': None, 'project_id': '58a61fc3992944ce971404a2ece6ff98', 'revision_number': 4, 'admin_state_up': True, 'shared': False, 'mtu': 1450, 'id': 'net1234'}
    self.find_mock.return_value = 'net1234'
    self.create_mock.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.subnet_show_mock.return_value = {'subnet': subnet_dict}
    self.network_show_mock.return_value = {'network': network_dict}
    self.port_show_mock.return_value = {'port': {'status': 'DOWN', 'name': utils.PhysName(stack.name, 'port'), 'allowed_address_pairs': [], 'admin_state_up': True, 'network_id': 'net1234', 'device_id': 'dc68eg2c-b60g-4b3f-bd82-67ec89280532', 'mac_address': 'fa:16:3e:75:67:60', 'tenant_id': '58a61fc3992944ce971404a2ece6ff98', 'security_groups': ['5b15d80c-6b70-4a1c-89c9-253538c5ade6'], 'fixed_ips': [{'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.0.2'}]}}
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.create_mock.assert_called_once_with({'port': {'network_id': u'net1234', 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'device_owner': u'network:dhcp', 'binding:vnic_type': 'normal', 'device_id': ''}})
    self.assertEqual('DOWN', port.FnGetAtt('status'))
    self.assertEqual([], port.FnGetAtt('allowed_address_pairs'))
    self.assertTrue(port.FnGetAtt('admin_state_up'))
    self.assertEqual('net1234', port.FnGetAtt('network_id'))
    self.assertEqual('fa:16:3e:75:67:60', port.FnGetAtt('mac_address'))
    self.assertEqual(utils.PhysName(stack.name, 'port'), port.FnGetAtt('name'))
    self.assertEqual('dc68eg2c-b60g-4b3f-bd82-67ec89280532', port.FnGetAtt('device_id'))
    self.assertEqual('58a61fc3992944ce971404a2ece6ff98', port.FnGetAtt('tenant_id'))
    self.assertEqual(['5b15d80c-6b70-4a1c-89c9-253538c5ade6'], port.FnGetAtt('security_groups'))
    self.assertEqual([{'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.0.2'}], port.FnGetAtt('fixed_ips'))
    self.assertEqual([subnet_dict], port.FnGetAtt('subnets'))
    self.assertEqual(network_dict, port.FnGetAtt('network'))
    self.assertRaises(exception.InvalidTemplateAttribute, port.FnGetAtt, 'Foo')