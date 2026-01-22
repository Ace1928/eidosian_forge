from unittest import mock
from openstack.database.v1 import instance
from openstack.tests.unit import base
def test_action_resize_volume(self):
    sot = instance.Instance(**EXAMPLE)
    response = mock.Mock()
    response.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    size = 4
    self.assertIsNone(sot.resize_volume(sess, size))
    url = 'instances/%s/action' % IDENTIFIER
    body = {'resize': {'volume': size}}
    sess.post.assert_called_with(url, json=body)