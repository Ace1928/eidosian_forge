import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_load_discovery_override_by_endpoints(self):
    self.requests_mock.get(self.DISCOVERY_URL, json=oidc_fixtures.DISCOVERY_DOCUMENT)
    access_token_endpoint = uuid.uuid4().hex
    plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint=self.DISCOVERY_URL, access_token_endpoint=access_token_endpoint)
    self.assertEqual(access_token_endpoint, plugin._get_access_token_endpoint(self.session))