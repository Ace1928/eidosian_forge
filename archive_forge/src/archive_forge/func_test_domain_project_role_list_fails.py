import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_domain_project_role_list_fails(self):
    user_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    self.assertRaises(exceptions.ValidationError, self.manager.list, domain=domain_id, project=project_id, user=user_id)