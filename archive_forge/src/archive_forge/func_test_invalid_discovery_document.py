import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_invalid_discovery_document(self):
    self.requests_mock.get(self.DISCOVERY_URL, json={})
    plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL)
    self.assertRaises(exceptions.InvalidOidcDiscoveryDocument, plugin._get_discovery_document, self.session)