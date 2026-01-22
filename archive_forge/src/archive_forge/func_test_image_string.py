from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(meta, 'get_server_external_ipv4')
def test_image_string(self, mock_get_server_external_ipv4):
    mock_get_server_external_ipv4.return_value = PUBLIC_V4
    server = standard_fake_server
    server['image'] = 'fake-image-id'
    hostvars = meta.get_hostvars_from_server(FakeCloud(), meta.obj_to_munch(server))
    self.assertEqual('fake-image-id', hostvars['image']['id'])