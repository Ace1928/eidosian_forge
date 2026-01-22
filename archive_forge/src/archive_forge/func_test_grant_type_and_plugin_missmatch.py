import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_grant_type_and_plugin_missmatch(self):
    self.assertRaises(exceptions.OidcGrantTypeMissmatch, self.plugin.__class__, self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, grant_type=uuid.uuid4().hex)