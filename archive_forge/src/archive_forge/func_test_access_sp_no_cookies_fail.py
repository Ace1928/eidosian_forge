import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_access_sp_no_cookies_fail(self):
    self.session.session.cookies = []
    self.assertRaises(exceptions.AuthorizationFailure, self.adfsplugin._access_service_provider, self.session)