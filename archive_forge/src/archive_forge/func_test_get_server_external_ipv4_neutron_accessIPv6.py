from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_external_ipv4_neutron_accessIPv6(self):
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE')
    srv['accessIPv6'] = PUBLIC_V6
    ip = meta.get_server_external_ipv6(server=srv)
    self.assertEqual(PUBLIC_V6, ip)