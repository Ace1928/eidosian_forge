import base64
import uuid
import requests
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1 import fixture as ksa_fixtures
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_initial_sp_call_invalid_response(self):
    """Send initial SP HTTP request and receive wrong server response."""
    self.requests_mock.get(self.default_sp_url, headers=CONTENT_TYPE_PAOS_HEADER, text='NON XML RESPONSE')
    self.assertRaises(exceptions.AuthorizationFailure, self.get_plugin().get_auth_ref, self.session)
    self.assertEqual(self.calls, [self.default_sp_url])