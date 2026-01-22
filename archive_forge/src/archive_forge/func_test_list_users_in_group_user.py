import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_users_in_group_user(self):
    group = self.create_group()
    user = self.create_user()
    self.driver.add_user_to_group(user['id'], group['id'])
    users = self.driver.list_users_in_group(group['id'], driver_hints.Hints())
    self.assertEqual([user['id']], [u['id'] for u in users])