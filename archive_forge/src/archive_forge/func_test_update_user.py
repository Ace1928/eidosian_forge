import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_user(self):
    user = self.create_user()
    user_mod = {'enabled': False}
    actual_user = self.driver.update_user(user['id'], user_mod)
    self.assertEqual(user['id'], actual_user['id'])
    self.assertIs(False, actual_user['enabled'])