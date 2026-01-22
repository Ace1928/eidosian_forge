from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_ip(self):
    srv = meta.obj_to_munch(standard_fake_server)
    self.assertEqual(PRIVATE_V4, meta.get_server_ip(srv, ext_tag='fixed'))
    self.assertEqual(PUBLIC_V4, meta.get_server_ip(srv, ext_tag='floating'))