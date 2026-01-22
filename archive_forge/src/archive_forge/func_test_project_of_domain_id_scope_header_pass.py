import uuid
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import tokenless_auth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_project_of_domain_id_scope_header_pass(self):
    project_name = uuid.uuid4().hex
    project_domain_id = uuid.uuid4().hex
    auth, session = self.create(auth_url=self.TEST_URL, project_name=project_name, project_domain_id=project_domain_id)
    session.get(self.TEST_URL, authenticated=True)
    self.assertRequestHeaderEqual('X-Project-Name', project_name)
    self.assertRequestHeaderEqual('X-Project-Domain-Id', project_domain_id)