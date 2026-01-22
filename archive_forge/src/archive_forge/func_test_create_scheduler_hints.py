import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import volume
from openstack import exceptions
from openstack.tests.unit import base
def test_create_scheduler_hints(self):
    sot = volume.Volume(**VOLUME)
    sot._translate_response = mock.Mock()
    sot.create(self.sess)
    url = '/volumes'
    volume_body = copy.deepcopy(VOLUME)
    scheduler_hints = volume_body.pop('OS-SCH-HNT:scheduler_hints')
    body = {'volume': volume_body, 'OS-SCH-HNT:scheduler_hints': scheduler_hints}
    self.sess.post.assert_called_with(url, json=body, microversion='3.0', headers={}, params={})