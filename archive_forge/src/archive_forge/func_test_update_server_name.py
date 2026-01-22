import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_server_name(self):
    """
        Test that update_server updates the name without raising any exception
        """
    fake_update_server = fakes.make_fake_server(self.server_id, self.updated_server_name)
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=%s' % self.server_name]), json={'servers': [self.fake_server]}), dict(method='PUT', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), json={'server': fake_update_server}, validate=dict(json={'server': {'name': self.updated_server_name}})), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []})])
    self.assertEqual(self.updated_server_name, self.cloud.update_server(self.server_name, name=self.updated_server_name)['name'])
    self.assert_calls()