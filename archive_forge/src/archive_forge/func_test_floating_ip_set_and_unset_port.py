import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_floating_ip_set_and_unset_port(self):
    """Test Floating IP Set and Unset port"""
    ext_subnet_id = self._create_subnet(self.EXTERNAL_NETWORK_NAME, 'ext-test-delete')
    self.addCleanup(self.openstack, 'subnet delete ' + ext_subnet_id)
    priv_subnet_id = self._create_subnet(self.PRIVATE_NETWORK_NAME, 'priv-test-delete')
    self.addCleanup(self.openstack, 'subnet delete ' + priv_subnet_id)
    self.ROUTER = uuid.uuid4().hex
    self.PORT_NAME = uuid.uuid4().hex
    json_output = self.openstack('floating ip create ' + '--description aaaa ' + self.EXTERNAL_NETWORK_NAME, parse_output=True)
    self.assertIsNotNone(json_output['id'])
    ip1 = json_output['id']
    self.addCleanup(self.openstack, 'floating ip delete ' + ip1)
    self.assertEqual('aaaa', json_output['description'])
    json_output = self.openstack('port create ' + '--network ' + self.PRIVATE_NETWORK_NAME + ' ' + '--fixed-ip subnet=' + priv_subnet_id + ' ' + self.PORT_NAME, parse_output=True)
    self.assertIsNotNone(json_output['id'])
    port_id = json_output['id']
    json_output = self.openstack('router create ' + self.ROUTER, parse_output=True)
    self.assertIsNotNone(json_output['id'])
    self.addCleanup(self.openstack, 'router delete ' + self.ROUTER)
    self.openstack('router add port ' + self.ROUTER + ' ' + port_id)
    self.openstack('router set ' + '--external-gateway ' + self.EXTERNAL_NETWORK_NAME + ' ' + self.ROUTER)
    self.addCleanup(self.openstack, 'router unset --external-gateway ' + self.ROUTER)
    self.addCleanup(self.openstack, 'router remove port ' + self.ROUTER + ' ' + port_id)
    self.openstack('floating ip set ' + '--port ' + port_id + ' ' + ip1)
    self.addCleanup(self.openstack, 'floating ip unset --port ' + ip1)
    json_output = self.openstack('floating ip show ' + ip1, parse_output=True)
    self.assertEqual(port_id, json_output['port_id'])