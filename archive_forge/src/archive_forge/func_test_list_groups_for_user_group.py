import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_groups_for_user_group(self):
    user = self.create_user()
    group = self.create_group()
    self.driver.add_user_to_group(user['id'], group['id'])
    groups = self.driver.list_groups_for_user(user['id'], driver_hints.Hints())
    self.assertEqual([group['id']], [g['id'] for g in groups])