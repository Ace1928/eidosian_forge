from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_unpause(self):
    sot = server.Server(**EXAMPLE)
    res = sot.unpause(self.sess)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'unpause': None}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)