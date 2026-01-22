import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_no_wait(self):
    """
        Test that create_server with no wait and no exception in the
        create call returns the server instance.
        """
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': fake_server})])
    self.assertDictEqual(server.Server(**fake_server).to_dict(computed=False), self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id')).to_dict(computed=False))
    self.assert_calls()