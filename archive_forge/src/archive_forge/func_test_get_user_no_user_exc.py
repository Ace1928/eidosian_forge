import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_user_no_user_exc(self):
    self.assertRaises(exception.UserNotFound, self.driver.get_user, uuid.uuid4().hex)