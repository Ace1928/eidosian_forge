import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_update_idp_authorization_ttl(self):
    body = self.default_body.copy()
    body['authorization_ttl'] = 10080
    default_resp = self._create_default_idp(body=body)
    default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
    idp_id = default_idp.get('id')
    url = self.base_url(suffix=idp_id)
    self.assertIsNotNone(idp_id)
    body['authorization_ttl'] = None
    body = {'identity_provider': body}
    resp = self.patch(url, body=body)
    updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
    body = body['identity_provider']
    self.assertEqual(body['authorization_ttl'], updated_idp.get('authorization_ttl'))
    resp = self.get(url)
    returned_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
    self.assertEqual(body['authorization_ttl'], returned_idp.get('authorization_ttl'))