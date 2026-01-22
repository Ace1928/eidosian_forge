import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_delete_router_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers', self.router_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers'], qs_elements=['name=%s' % self.router_name]), json={'routers': []})])
    self.assertFalse(self.cloud.delete_router(self.router_name))
    self.assert_calls()