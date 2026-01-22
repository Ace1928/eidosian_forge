from castellan.common.credentials import keystone_password
from castellan.tests import base
def test_get_reauthenticate(self):
    self.assertEqual(self.reauthenticate, self.ks_password_credential.reauthenticate)