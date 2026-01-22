import uuid
from keystone.common import driver_hints
from keystone import exception
def test_check_user_in_group_user_not_in_group_exc(self):
    user = self.create_user()
    group = self.create_group()
    self.assertRaises(exception.NotFound, self.driver.check_user_in_group, user['id'], group['id'])