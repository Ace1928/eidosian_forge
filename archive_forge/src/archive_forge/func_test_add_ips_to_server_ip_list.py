from unittest.mock import patch
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@patch.object(connection.Connection, 'add_ip_list')
def test_add_ips_to_server_ip_list(self, mock_add_ip_list):
    server_dict = fakes.make_fake_server(server_id='server-id', name='test-server', status='ACTIVE', addresses={})
    ips = ['203.0.113.29', '172.24.4.229']
    self.cloud.add_ips_to_server(server_dict, ips=ips)
    mock_add_ip_list.assert_called_with(server_dict, ips, wait=False, timeout=60, fixed_address=None, nat_destination=None)