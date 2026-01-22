import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_send_assertion_to_service_provider_bad_status(self):
    self.requests_mock.register_uri('POST', self.SP_ENDPOINT, status_code=500)
    self.adfsplugin.adfs_token = etree.XML(self.ADFS_SECURITY_TOKEN_RESPONSE)
    self.adfsplugin._prepare_sp_request()
    self.assertRaises(exceptions.InternalServerError, self.adfsplugin._send_assertion_to_service_provider, self.session)