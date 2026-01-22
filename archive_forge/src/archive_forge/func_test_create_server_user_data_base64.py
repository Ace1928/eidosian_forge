import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_user_data_base64(self):
    """
        Test that a server passed user-data sends it base64 encoded.
        """
    user_data = self.getUniqueString('user_data')
    user_data_b64 = base64.b64encode(user_data.encode('utf-8')).decode('utf-8')
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    fake_server['user_data'] = user_data
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'user_data': user_data_b64, 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': fake_server})])
    self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id'), userdata=user_data, wait=False)
    self.assert_calls()