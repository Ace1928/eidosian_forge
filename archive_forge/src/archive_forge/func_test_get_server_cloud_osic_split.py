from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(connection.Connection, 'get_volumes')
@mock.patch.object(connection.Connection, 'get_image_name')
@mock.patch.object(connection.Connection, 'get_flavor_name')
def test_get_server_cloud_osic_split(self, mock_get_flavor_name, mock_get_image_name, mock_get_volumes):
    self.cloud._floating_ip_source = None
    self.cloud.force_ipv4 = False
    self.cloud._local_ipv6 = True
    self.cloud._external_ipv4_names = ['GATEWAY_NET']
    self.cloud._external_ipv6_names = ['GATEWAY_NET_V6']
    self.cloud._internal_ipv4_names = ['GATEWAY_NET_V6']
    self.cloud._internal_ipv6_names = []
    mock_get_image_name.return_value = 'cirros-0.3.4-x86_64-uec'
    mock_get_flavor_name.return_value = 'm1.tiny'
    mock_get_volumes.return_value = []
    fake_server = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', flavor={u'id': u'1'}, image={'name': u'cirros-0.3.4-x86_64-uec', u'id': u'f93d000b-7c29-4489-b375-3641a1758fe1'}, addresses={'private': [{'addr': '10.223.160.141', 'version': 4}], 'public': [{'addr': '104.130.246.91', 'version': 4}, {'addr': '2001:4800:7819:103:be76:4eff:fe05:8525', 'version': 6}]})
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': OSIC_NETWORKS}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': OSIC_SUBNETS}), self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', fake_server['id']]), json=fake_server), dict(method='GET', uri='{endpoint}/servers/test-id/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': []})])
    srv = self.cloud.get_openstack_vars(_server.Server(**fake_server))
    self.assertEqual('10.223.160.141', srv['private_v4'])
    self.assertEqual('104.130.246.91', srv['public_v4'])
    self.assertEqual('2001:4800:7819:103:be76:4eff:fe05:8525', srv['public_v6'])
    self.assertEqual('2001:4800:7819:103:be76:4eff:fe05:8525', srv['interface_ip'])
    self.assert_calls()