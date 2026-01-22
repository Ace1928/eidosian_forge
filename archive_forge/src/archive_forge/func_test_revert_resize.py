from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_revert_resize(self):
    sot = server.Server(**EXAMPLE)
    self.assertIsNone(sot.revert_resize(self.sess))
    url = 'servers/IDENTIFIER/action'
    body = {'revertResize': None}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)