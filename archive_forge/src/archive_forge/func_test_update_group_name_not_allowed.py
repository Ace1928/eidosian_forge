import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_group_name_not_allowed(self):
    if self.allows_name_update:
        self.skipTest('driver allows name update')
    group = self.create_group()
    group_mod = {'name': uuid.uuid4().hex}
    self.assertRaises(exception.ValidationError, self.driver.update_group, group['id'], group_mod)