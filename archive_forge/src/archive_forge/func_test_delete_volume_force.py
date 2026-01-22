import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_volume_force(self):
    vol = {'id': 'volume001', 'status': 'attached', 'name': '', 'attachments': []}
    volume = meta.obj_to_munch(fakes.FakeVolume(**vol))
    self.register_uris([self.get_cinder_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['volumes', volume['id']]), json={'volumes': [volume]}), dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['volumes', volume.id, 'action']), validate=dict(json={'os-force_delete': None})), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['volumes', volume['id']]), status_code=404)])
    self.assertTrue(self.cloud.delete_volume(volume['id'], force=True))
    self.assert_calls()