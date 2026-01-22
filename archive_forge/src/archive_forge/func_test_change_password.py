import uuid
from keystone.common import driver_hints
from keystone import exception
def test_change_password(self):
    if not self.allows_self_service_change_password:
        self.skipTest("Backend doesn't allow change password.")
    password = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    user = self.create_user(domain_id=domain_id, password=password)
    new_password = uuid.uuid4().hex
    self.driver.change_password(user['id'], new_password)
    self.driver.authenticate(user['id'], new_password)