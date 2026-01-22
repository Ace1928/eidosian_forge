import uuid
from keystone.common import driver_hints
from keystone import exception
def test_add_user_to_group_no_user_exc(self):
    group = self.create_group()
    user_id = uuid.uuid4().hex
    self.assertRaises(exception.UserNotFound, self.driver.add_user_to_group, user_id, group['id'])