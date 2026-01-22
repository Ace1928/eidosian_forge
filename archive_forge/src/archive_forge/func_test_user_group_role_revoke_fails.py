import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_user_group_role_revoke_fails(self):
    user_id = uuid.uuid4().hex
    group_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    ref = self.new_ref()
    self.assertRaises(exceptions.ValidationError, self.manager.revoke, role=ref['id'], project=project_id, group=group_id, user=user_id)