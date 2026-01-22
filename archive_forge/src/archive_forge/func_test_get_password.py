from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_password(self):
    self.assertEqual(self.password, self.ks_password_credential.password)