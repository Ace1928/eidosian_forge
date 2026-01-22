import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_group(self):
    group = self.create_group()
    actual_group = self.driver.get_group(group['id'])
    self.assertEqual(group['id'], actual_group['id'])