from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import volume
from openstack.tests.unit import base
def test_reset_status(self):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.reset_status(self.sess, '1', '2', '3'))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-reset_status': {'status': '1', 'attach_status': '2', 'migration_status': '3'}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)