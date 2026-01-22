import uuid
from keystone.common import driver_hints
from keystone import exception
def test_list_users_in_group_no_group(self):
    group_id = uuid.uuid4().hex
    self.assertRaises(exception.GroupNotFound, self.driver.list_users_in_group, group_id, driver_hints.Hints())