import uuid
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import tokenless_auth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_no_scope_header_fail(self):
    auth, session = self.create(auth_url=self.TEST_URL)
    self.assertIsNone(auth.get_headers(session))
    msg = 'No valid authentication is available'
    self.assertRaisesRegex(exceptions.AuthorizationFailure, msg, session.get, self.TEST_URL, authenticated=True)