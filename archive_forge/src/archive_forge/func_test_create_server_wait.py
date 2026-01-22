import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(connection.Connection, 'wait_for_server')
def test_create_server_wait(self, mock_wait):
    """
        Test that create_server with a wait actually does the wait.
        """
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}}))])
    (self.cloud.create_server('server-name', dict(id='image-id'), dict(id='flavor-id'), wait=True),)
    srv = server.Server.existing(connection=self.cloud, min_count=1, max_count=1, networks='auto', imageRef='image-id', flavorRef='flavor-id', **fake_server)
    mock_wait.assert_called_once_with(srv, auto_ip=True, ips=None, ip_pool=None, reuse=True, timeout=180, nat_destination=None)
    self.assert_calls()