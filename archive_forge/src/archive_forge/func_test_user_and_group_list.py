from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_user_and_group_list(self):
    self.assertRaises(exceptions.ValidationError, self.manager.list, user=self.TEST_USER_ID, group=self.TEST_GROUP_ID)