from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_create_port_parameters(self):
    """Test that we detect invalid arguments passed to create_port"""
    self.assertRaises(TypeError, self.cloud.create_port, network_id='test-net-id', nome='test-port-name', stato_amministrativo_porta=True)