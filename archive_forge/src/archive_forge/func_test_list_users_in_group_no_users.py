import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_users_in_group_no_users(self):
    group = self.create_group()
    users = self.driver.list_users_in_group(group['id'], driver_hints.Hints())
    self.assertEqual([], users)