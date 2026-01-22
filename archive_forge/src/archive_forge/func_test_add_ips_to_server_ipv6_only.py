from unittest.mock import patch
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@patch.object(connection.Connection, 'has_service')
@patch.object(connection.Connection, 'get_floating_ip')
@patch.object(connection.Connection, '_add_auto_ip')
def test_add_ips_to_server_ipv6_only(self, mock_add_auto_ip, mock_get_floating_ip, mock_has_service):
    self.cloud._floating_ip_source = None
    self.cloud.force_ipv4 = False
    self.cloud._local_ipv6 = True
    mock_has_service.return_value = False
    server = fakes.make_fake_server(server_id='server-id', name='test-server', status='ACTIVE', addresses={'private': [{'addr': '10.223.160.141', 'version': 4}], 'public': [{u'OS-EXT-IPS-MAC:mac_addr': u'fa:16:3e:ae:7d:42', u'OS-EXT-IPS:type': u'fixed', 'addr': '2001:4800:7819:103:be76:4eff:fe05:8525', 'version': 6}]})
    server_dict = meta.add_server_interfaces(self.cloud, _server.Server(**server))
    new_server = self.cloud.add_ips_to_server(server=server_dict)
    mock_get_floating_ip.assert_not_called()
    mock_add_auto_ip.assert_not_called()
    self.assertEqual(new_server['interface_ip'], '2001:4800:7819:103:be76:4eff:fe05:8525')
    self.assertEqual(new_server['private_v4'], '10.223.160.141')
    self.assertEqual(new_server['public_v4'], '')
    self.assertEqual(new_server['public_v6'], '2001:4800:7819:103:be76:4eff:fe05:8525')