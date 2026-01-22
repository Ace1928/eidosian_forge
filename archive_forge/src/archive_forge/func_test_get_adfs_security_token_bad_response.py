import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_get_adfs_security_token_bad_response(self):
    """Test proper handling HTTP 500 and mangled (non XML) response.

        This should never happen yet, keystoneauth1 should be prepared
        and correctly raise exceptions.InternalServerError once it cannot
        parse XML fault message
        """
    self.requests_mock.register_uri('POST', self.IDENTITY_PROVIDER_URL, content=b'NOT XML', status_code=500)
    self.adfsplugin._prepare_adfs_request()
    self.assertRaises(exceptions.InternalServerError, self.adfsplugin._get_adfs_security_token, self.session)