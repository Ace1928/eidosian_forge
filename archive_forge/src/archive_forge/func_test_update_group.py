import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_group(self):
    group = self.create_group()
    new_description = uuid.uuid4().hex
    group_mod = {'description': new_description}
    actual_group = self.driver.update_group(group['id'], group_mod)
    self.assertEqual(new_description, actual_group['description'])