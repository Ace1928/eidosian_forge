from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_external_ipv6(self):
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', addresses={'test-net': [{'addr': PUBLIC_V4, 'version': 4}, {'addr': PUBLIC_V6, 'version': 6}]})
    ip = meta.get_server_external_ipv6(srv)
    self.assertEqual(PUBLIC_V6, ip)