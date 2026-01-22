from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_reboot(self):
    sot = server.Server(**EXAMPLE)
    self.assertIsNone(sot.reboot(self.sess, 'HARD'))
    url = 'servers/IDENTIFIER/action'
    body = {'reboot': {'type': 'HARD'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)