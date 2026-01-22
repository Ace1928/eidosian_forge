import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_load_discovery(self):
    self.requests_mock.get(self.DISCOVERY_URL, json=oidc_fixtures.DISCOVERY_DOCUMENT)
    plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL)
    self.assertEqual(oidc_fixtures.DISCOVERY_DOCUMENT['token_endpoint'], plugin._get_access_token_endpoint(self.session))