import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_network_with_empty_nics(self):
    """
        Verify that if 'network' is supplied, along with an empty 'nics' list,
        it's treated the same as if 'nics' were not included.
        """
    network = {'id': 'network-id', 'name': 'network-name'}
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', 'network-name']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=network-name']), json={'networks': [network]}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'networks': [{'uuid': 'network-id'}], 'name': 'server-name'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': build_server}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': [network]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets']), json={'subnets': []})])
    self.cloud.create_server('server-name', dict(id='image-id'), dict(id='flavor-id'), network='network-name', nics=[])
    self.assert_calls()