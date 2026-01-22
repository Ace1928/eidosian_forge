from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import volume
from openstack.tests.unit import base
def test_detach(self):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.detach(self.sess, '1'))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-detach': {'attachment_id': '1'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)