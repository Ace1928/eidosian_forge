from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_private_ip(self):
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': [{'id': 'test-net-id', 'name': 'test-net-name'}]}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': SUBNETS_WITH_NAT})])
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', addresses={'private': [{'OS-EXT-IPS:type': 'fixed', 'addr': PRIVATE_V4, 'version': 4}], 'public': [{'OS-EXT-IPS:type': 'floating', 'addr': PUBLIC_V4, 'version': 4}]})
    self.assertEqual(PRIVATE_V4, meta.get_server_private_ip(srv, self.cloud))
    self.assert_calls()