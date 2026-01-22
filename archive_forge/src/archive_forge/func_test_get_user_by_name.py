import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_user_by_name(self):
    domain_id = uuid.uuid4().hex
    user = self.create_user(domain_id=domain_id)
    actual_user = self.driver.get_user_by_name(user['name'], domain_id)
    self.assertEqual(user['id'], actual_user['id'])