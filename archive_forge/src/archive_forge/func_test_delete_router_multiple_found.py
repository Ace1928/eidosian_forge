import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_delete_router_multiple_found(self):
    router1 = dict(id='123', name='mickey')
    router2 = dict(id='456', name='mickey')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers', 'mickey']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers'], qs_elements=['name=mickey']), json={'routers': [router1, router2]})])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_router, 'mickey')
    self.assert_calls()