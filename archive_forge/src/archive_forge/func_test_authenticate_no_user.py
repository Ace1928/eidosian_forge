import uuid
from keystone.common import driver_hints
from keystone import exception
def test_authenticate_no_user(self):
    user_id = uuid.uuid4().hex
    password = uuid.uuid4().hex
    self.assertRaises(AssertionError, self.driver.authenticate, user_id, password)