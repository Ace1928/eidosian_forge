import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_boot_attach_volume(self):
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    active_server = fakes.make_fake_server('1234', '', 'BUILD')
    vol = {'id': 'volume001', 'status': 'available', 'name': '', 'attachments': []}
    volume = meta.obj_to_munch(fakes.FakeVolume(**vol))
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'block_device_mapping_v2': [{'boot_index': 0, 'delete_on_termination': True, 'destination_type': 'local', 'source_type': 'image', 'uuid': 'image-id'}, {'boot_index': '-1', 'delete_on_termination': False, 'destination_type': 'volume', 'source_type': 'volume', 'uuid': 'volume001'}], 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': active_server})])
    self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id'), boot_from_volume=False, volumes=[volume], wait=False)
    self.assert_calls()