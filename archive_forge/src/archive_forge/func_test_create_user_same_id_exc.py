import uuid
from keystone.common import driver_hints
from keystone import exception
def test_create_user_same_id_exc(self):
    user_id = uuid.uuid4().hex
    user = {'id': user_id, 'name': uuid.uuid4().hex, 'enabled': True}
    if self.driver.is_domain_aware():
        user['domain_id'] = uuid.uuid4().hex
    self.driver.create_user(user_id, user)
    self.assertRaises(exception.Conflict, self.driver.create_user, user_id, user)