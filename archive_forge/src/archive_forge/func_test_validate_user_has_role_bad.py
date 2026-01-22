from unittest import mock
from keystoneauth1 import adapter
from openstack.identity.v3 import domain
from openstack.identity.v3 import group
from openstack.identity.v3 import role
from openstack.identity.v3 import user
from openstack.tests.unit import base
def test_validate_user_has_role_bad(self):
    sot = domain.Domain(**EXAMPLE)
    resp = self.bad_resp
    self.sess.head = mock.Mock(return_value=resp)
    self.assertFalse(sot.validate_user_has_role(self.sess, user.User(id='1'), role.Role(id='2')))