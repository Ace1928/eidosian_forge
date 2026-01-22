import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_attach_volume_exception(self):
    server = dict(id='server001')
    vol = {'id': 'volume001', 'status': 'available', 'name': '', 'attachments': []}
    volume = meta.obj_to_munch(fakes.FakeVolume(**vol))
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers', server['id'], 'os-volume_attachments']), status_code=404, validate=dict(json={'volumeAttachment': {'volumeId': vol['id']}}))])
    with testtools.ExpectedException(exceptions.NotFoundException):
        self.cloud.attach_volume(server, volume, wait=False)
    self.assert_calls()