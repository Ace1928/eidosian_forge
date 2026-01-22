from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_backup(self):
    sot = server.Server(**EXAMPLE)
    res = sot.backup(self.sess, 'name', 'daily', 1)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'createBackup': {'name': 'name', 'backup_type': 'daily', 'rotation': 1}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)