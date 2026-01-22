from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_username(self):
    self.assertEqual(self.username, self.ks_password_credential.username)