from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_az(self):
    server = standard_fake_server
    server['OS-EXT-AZ:availability_zone'] = 'az1'
    hostvars = self.cloud._normalize_server(meta.obj_to_munch(server))
    self.assertEqual('az1', hostvars['az'])