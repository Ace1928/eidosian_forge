import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_flavor_show(self):
    """Test show network flavor"""
    name = uuid.uuid4().hex
    cmd_output = self.openstack('network flavor create --description testdescription --disable --service-type  L3_ROUTER_NAT ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'network flavor delete ' + name)
    cmd_output = self.openstack('network flavor show ' + name, parse_output=True)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual(False, cmd_output['enabled'])
    self.assertEqual('testdescription', cmd_output['description'])