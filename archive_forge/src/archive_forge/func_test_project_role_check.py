import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_project_role_check(self):
    user_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    ref = self.new_ref()
    self.stub_url('HEAD', ['projects', project_id, 'users', user_id, self.collection_key, ref['id']], status_code=200)
    self.manager.check(role=ref['id'], project=project_id, user=user_id)