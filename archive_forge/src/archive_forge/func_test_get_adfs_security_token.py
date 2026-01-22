import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_get_adfs_security_token(self):
    """Test ADFSPassword._get_adfs_security_token()."""
    self.requests_mock.post(self.IDENTITY_PROVIDER_URL, content=utils.make_oneline(self.ADFS_SECURITY_TOKEN_RESPONSE), status_code=200)
    self.adfsplugin._prepare_adfs_request()
    self.adfsplugin._get_adfs_security_token(self.session)
    adfs_response = etree.tostring(self.adfsplugin.adfs_token)
    fixture_response = self.ADFS_SECURITY_TOKEN_RESPONSE
    self.assertThat(fixture_response, matchers.XMLEquals(adfs_response))