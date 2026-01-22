import os
import urllib.parse
import uuid
from lxml import etree
from oslo_config import fixture as config
import requests
from keystoneclient.auth import conf
from keystoneclient.contrib.auth.v3 import saml2
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import saml2_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib.federation import saml as saml_manager
def test_initial_sp_call(self):
    """Test initial call, expect SOAP message."""
    self.requests_mock.get(self.FEDERATION_AUTH_URL, content=make_oneline(saml2_fixtures.SP_SOAP_RESPONSE))
    a = self.saml2plugin._send_service_provider_request(self.session)
    self.assertFalse(a)
    fixture_soap_response = make_oneline(saml2_fixtures.SP_SOAP_RESPONSE)
    sp_soap_response = make_oneline(etree.tostring(self.saml2plugin.saml2_authn_request))
    error_msg = 'Expected %s instead of %s' % (fixture_soap_response, sp_soap_response)
    self.assertEqual(fixture_soap_response, sp_soap_response, error_msg)
    self.assertEqual(self.saml2plugin.sp_response_consumer_url, self.SHIB_CONSUMER_URL, 'Expected consumer_url set to %s instead of %s' % (self.SHIB_CONSUMER_URL, str(self.saml2plugin.sp_response_consumer_url)))