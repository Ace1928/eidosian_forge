import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_servers_exception(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), status_code=400)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_servers)
    self.assert_calls()