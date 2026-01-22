from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_internal_provider_ipv4_neutron(self):
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': [{'id': 'test-net-id', 'name': 'test-net', 'router:external': False, 'provider:network_type': 'vxlan', 'provider:physical_network': None}]}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': SUBNETS_WITH_NAT})])
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', addresses={'test-net': [{'addr': PRIVATE_V4, 'version': 4}]})
    self.assertIsNone(meta.get_server_external_ipv4(cloud=self.cloud, server=srv))
    int_ip = meta.get_server_private_ip(cloud=self.cloud, server=srv)
    self.assertEqual(PRIVATE_V4, int_ip)
    self.assert_calls()