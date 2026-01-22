import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_create_router(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers']), json={'router': self.mock_router_rep}, validate=dict(json={'router': {'name': self.router_name, 'admin_state_up': True}}))])
    new_router = self.cloud.create_router(name=self.router_name, admin_state_up=True)
    self._compare_routers(self.mock_router_rep, new_router)
    self.assert_calls()