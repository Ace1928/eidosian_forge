from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_rebuild_minimal(self):
    sot = server.Server(**EXAMPLE)
    sot._translate_response = lambda arg: arg
    result = sot.rebuild(self.sess, name='nootoo', admin_password='seekr3two', image='http://image/2')
    self.assertIsInstance(result, server.Server)
    url = 'servers/IDENTIFIER/action'
    body = {'rebuild': {'name': 'nootoo', 'imageRef': 'http://image/2', 'adminPass': 'seekr3two'}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)