from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_has_no_volume_service(self):
    fake_cloud = FakeCloud()
    fake_cloud.service_val = False
    hostvars = meta.get_hostvars_from_server(fake_cloud, meta.obj_to_munch(standard_fake_server))
    self.assertEqual([], hostvars['volumes'])