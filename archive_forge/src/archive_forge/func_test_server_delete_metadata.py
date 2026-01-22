import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_server_delete_metadata(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [self.fake_server]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', self.fake_server['id'], 'metadata', 'key']), status_code=200)])
    self.cloud.delete_server_metadata(self.server_id, ['key'])
    self.assert_calls()