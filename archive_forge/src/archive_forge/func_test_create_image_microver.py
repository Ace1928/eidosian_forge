from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_create_image_microver(self):
    sot = server.Server(**EXAMPLE)
    name = 'noo'
    metadata = {'nu': 'image', 'created': 'today'}
    url = 'servers/IDENTIFIER/action'
    body = {'createImage': {'name': name, 'metadata': metadata}}
    headers = {'Accept': ''}
    rsp = mock.Mock()
    rsp.json.return_value = {'image_id': 'dummy3'}
    rsp.headers = {'Location': 'dummy/dummy2'}
    rsp.status_code = 200
    self.sess.post.return_value = rsp
    self.endpoint_data = mock.Mock(spec=['min_microversion', 'max_microversion'], min_microversion='2.1', max_microversion='2.56')
    self.sess.get_endpoint_data.return_value = self.endpoint_data
    self.sess.default_microversion = None
    image_id = sot.create_image(self.sess, name, metadata)
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion='2.45')
    self.assertEqual('dummy3', image_id)