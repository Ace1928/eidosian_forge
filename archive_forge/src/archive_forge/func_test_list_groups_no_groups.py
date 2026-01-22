import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_groups_no_groups(self):
    groups = self.driver.list_groups(driver_hints.Hints())
    self.assertEqual([], groups)