import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_rebuild_server_with_admin_pass_wait(self):
    """
        Test that a server with an admin_pass passed returns the password
        """
    password = self.getUniqueString('password')
    rebuild_server = self.rebuild_server.copy()
    rebuild_server['adminPass'] = password
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id, 'action']), json={'server': rebuild_server}, validate=dict(json={'rebuild': {'imageRef': 'a', 'adminPass': password}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), json={'server': self.rebuild_server}), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', self.server_id]), json={'server': self.fake_server}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []})])
    self.assertEqual(password, self.cloud.rebuild_server(self.fake_server['id'], 'a', admin_pass=password, wait=True)['adminPass'])
    self.assert_calls()