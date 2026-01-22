import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_attach_volume_already_attached(self):
    device_id = 'device001'
    server = dict(id='server001')
    volume = dict(id='volume001', attachments=[{'server_id': 'server001', 'device': device_id}])
    with testtools.ExpectedException(exceptions.SDKException, 'Volume %s already attached to server %s on device %s' % (volume['id'], server['id'], device_id)):
        self.cloud.attach_volume(server, volume)
    self.assertEqual(0, len(self.adapter.request_history))