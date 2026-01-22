import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_add_remove_router_interface(self):
    router = self._create_and_verify_advanced_router(external_cidr=u'10.3.3.0/24')
    net_name = self.network_prefix + '_intnet1'
    sub_name = self.subnet_prefix + '_intsub1'
    net = self.operator_cloud.create_network(name=net_name)
    sub = self.operator_cloud.create_subnet(net['id'], '10.4.4.0/24', subnet_name=sub_name, gateway_ip='10.4.4.1')
    iface = self.operator_cloud.add_router_interface(router, subnet_id=sub['id'])
    self.assertIsNone(self.operator_cloud.remove_router_interface(router, subnet_id=sub['id']))
    self.assertIsNotNone(iface)
    for key in ('id', 'subnet_id', 'port_id', 'tenant_id'):
        self.assertIn(key, iface)
    self.assertEqual(router['id'], iface['id'])
    self.assertEqual(sub['id'], iface['subnet_id'])