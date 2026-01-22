import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_create_router_with_enable_snat_True(self):
    """Send enable_snat when it is True."""
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers']), json={'router': self.mock_router_rep}, validate=dict(json={'router': {'name': self.router_name, 'admin_state_up': True, 'external_gateway_info': {'enable_snat': True}}}))])
    self.cloud.create_router(name=self.router_name, admin_state_up=True, enable_snat=True)
    self.assert_calls()