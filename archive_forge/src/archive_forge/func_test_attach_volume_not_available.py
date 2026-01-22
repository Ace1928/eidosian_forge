import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_attach_volume_not_available(self):
    server = dict(id='server001')
    volume = dict(id='volume001', status='error', attachments=[])
    with testtools.ExpectedException(exceptions.SDKException, "Volume %s is not available. Status is '%s'" % (volume['id'], volume['status'])):
        self.cloud.attach_volume(server, volume)
    self.assertEqual(0, len(self.adapter.request_history))