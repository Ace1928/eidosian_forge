from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_shelve_offload(self):
    sot = server.Server(**EXAMPLE)
    self.assertIsNone(sot.shelve_offload(self.sess))
    url = 'servers/IDENTIFIER/action'
    body = {'shelveOffload': None}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)