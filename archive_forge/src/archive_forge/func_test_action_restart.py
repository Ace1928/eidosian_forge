from unittest import mock
from openstack.database.v1 import instance
from openstack.tests.unit import base
def test_action_restart(self):
    sot = instance.Instance(**EXAMPLE)
    response = mock.Mock()
    response.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    self.assertIsNone(sot.restart(sess))
    url = 'instances/%s/action' % IDENTIFIER
    body = {'restart': None}
    sess.post.assert_called_with(url, json=body)