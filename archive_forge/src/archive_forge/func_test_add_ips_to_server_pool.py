from unittest.mock import patch
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@patch.object(connection.Connection, '_add_ip_from_pool')
def test_add_ips_to_server_pool(self, mock_add_ip_from_pool):
    server_dict = fakes.make_fake_server(server_id='romeo', name='test-server', status='ACTIVE', addresses={})
    pool = 'nova'
    self.cloud.add_ips_to_server(server_dict, ip_pool=pool)
    mock_add_ip_from_pool.assert_called_with(server_dict, pool, reuse=True, wait=False, timeout=60, fixed_address=None, nat_destination=None)