import uuid
from keystone.common import driver_hints
from keystone import exception
def test_update_user_remove_optional_attribute(self):
    user = self.create_user(default_project_id=uuid.uuid4().hex)
    self.assertIn('default_project_id', user)
    user_mod = {'default_project_id': None}
    actual_user = self.driver.update_user(user['id'], user_mod)
    self.assertNotIn('default_project_id', actual_user)