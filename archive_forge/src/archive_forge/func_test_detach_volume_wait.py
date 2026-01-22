import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_detach_volume_wait(self):
    server = dict(id='server001')
    attachments = [{'server_id': 'server001', 'device': 'device001'}]
    vol = {'id': 'volume001', 'status': 'attached', 'name': '', 'attachments': attachments}
    volume = meta.obj_to_munch(fakes.FakeVolume(**vol))
    vol['status'] = 'available'
    vol['attachments'] = []
    avail_volume = meta.obj_to_munch(fakes.FakeVolume(**vol))
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', server['id']]), json={'server': server}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', server['id'], 'os-volume_attachments', volume.id])), self.get_cinder_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['volumes', 'detail']), json={'volumes': [avail_volume]})])
    self.cloud.detach_volume(server, volume)
    self.assert_calls()