import uuid
from openstackclient.tests.functional.network.v2 import common
def test_router_add_remove_route(self):
    network_name = uuid.uuid4().hex
    subnet_name = uuid.uuid4().hex
    router_name = uuid.uuid4().hex
    self.openstack('network create %s' % network_name)
    self.addCleanup(self.openstack, 'network delete %s' % network_name)
    self.openstack('subnet create %s --network %s --subnet-range 10.0.0.0/24' % (subnet_name, network_name))
    self.openstack('router create %s' % router_name)
    self.addCleanup(self.openstack, 'router delete %s' % router_name)
    self.openstack('router add subnet %s %s' % (router_name, subnet_name))
    self.addCleanup(self.openstack, 'router remove subnet %s %s' % (router_name, subnet_name))
    out1 = (self.openstack('router add route %s --route destination=10.0.10.0/24,gateway=10.0.0.10' % router_name, parse_output=True),)
    self.assertEqual(1, len(out1[0]['routes']))
    self.addCleanup(self.openstack, 'router set %s --no-route' % router_name)
    out2 = (self.openstack('router add route %s --route destination=10.0.10.0/24,gateway=10.0.0.10 --route destination=10.0.11.0/24,gateway=10.0.0.11' % router_name, parse_output=True),)
    self.assertEqual(2, len(out2[0]['routes']))
    out3 = (self.openstack('router remove route %s --route destination=10.0.11.0/24,gateway=10.0.0.11 --route destination=10.0.12.0/24,gateway=10.0.0.12' % router_name, parse_output=True),)
    self.assertEqual(1, len(out3[0]['routes']))