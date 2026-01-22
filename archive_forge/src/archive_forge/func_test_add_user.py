from unittest import mock
from keystoneauth1 import adapter
from openstack.identity.v3 import group
from openstack.identity.v3 import user
from openstack.tests.unit import base
def test_add_user(self):
    sot = group.Group(**EXAMPLE)
    resp = self.good_resp
    self.sess.put = mock.Mock(return_value=resp)
    sot.add_user(self.sess, user.User(id='1'))
    self.sess.put.assert_called_with('groups/IDENTIFIER/users/1')