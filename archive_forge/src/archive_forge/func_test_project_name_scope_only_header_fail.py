import uuid
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import tokenless_auth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_project_name_scope_only_header_fail(self):
    project_name = uuid.uuid4().hex
    auth, session = self.create(auth_url=self.TEST_URL, project_name=project_name)
    self.assertIsNone(auth.get_headers(session))
    msg = 'No valid authentication is available'
    self.assertRaisesRegex(exceptions.AuthorizationFailure, msg, session.get, self.TEST_URL, authenticated=True)