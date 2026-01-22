import uuid
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_group(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-server-groups']), json={'server_group': self.fake_group}, validate=dict(json={'server_group': {'name': self.group_name, 'policies': self.policies}}))])
    self.cloud.create_server_group(name=self.group_name, policies=self.policies)
    self.assert_calls()