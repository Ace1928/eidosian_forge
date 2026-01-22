from unittest import mock
from openstack.database.v1 import instance
from openstack.tests.unit import base
def test_is_root_enabled(self):
    sot = instance.Instance(**EXAMPLE)
    response = mock.Mock()
    response.body = {'rootEnabled': True}
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.get = mock.Mock(return_value=response)
    self.assertTrue(sot.is_root_enabled(sess))
    url = 'instances/%s/root' % IDENTIFIER
    sess.get.assert_called_with(url)