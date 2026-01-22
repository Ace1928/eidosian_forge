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
def test_initial_sp_call_when_saml_authenticated(self):
    self.requests_mock.get(self.FEDERATION_AUTH_URL, json=saml2_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': saml2_fixtures.UNSCOPED_TOKEN_HEADER})
    a = self.saml2plugin._send_service_provider_request(self.session)
    self.assertTrue(a)
    self.assertEqual(saml2_fixtures.UNSCOPED_TOKEN['token'], self.saml2plugin.authenticated_response.json()['token'])
    self.assertEqual(saml2_fixtures.UNSCOPED_TOKEN_HEADER, self.saml2plugin.authenticated_response.headers['X-Subject-Token'])