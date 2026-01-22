from unittest import mock
from keystoneauth1 import adapter
from openstack.identity.v3 import group
from openstack.identity.v3 import user
from openstack.tests.unit import base
def test_remove_user(self):
    sot = group.Group(**EXAMPLE)
    resp = self.good_resp
    self.sess.delete = mock.Mock(return_value=resp)
    sot.remove_user(self.sess, user.User(id='1'))
    self.sess.delete.assert_called_with('groups/IDENTIFIER/users/1')