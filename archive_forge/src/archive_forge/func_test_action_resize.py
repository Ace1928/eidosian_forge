from unittest import mock
from openstack.database.v1 import instance
from openstack.tests.unit import base
def test_action_resize(self):
    sot = instance.Instance(**EXAMPLE)
    response = mock.Mock()
    response.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    flavor = 'http://flavor/flav'
    self.assertIsNone(sot.resize(sess, flavor))
    url = 'instances/%s/action' % IDENTIFIER
    body = {'resize': {'flavorRef': flavor}}
    sess.post.assert_called_with(url, json=body)