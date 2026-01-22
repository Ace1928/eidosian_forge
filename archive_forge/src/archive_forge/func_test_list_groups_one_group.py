import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_groups_one_group(self):
    group = self.create_group()
    groups = self.driver.list_groups(driver_hints.Hints())
    self.assertEqual(group['id'], groups[0]['id'])