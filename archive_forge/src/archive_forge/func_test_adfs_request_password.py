import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_adfs_request_password(self):
    self.adfsplugin._prepare_adfs_request()
    password = self.adfsplugin.prepared_request.xpath(self.PASSWORD_XPATH, namespaces=self.NAMESPACES)[0]
    self.assertEqual(self.TEST_TOKEN, password.text)