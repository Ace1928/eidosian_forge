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
def test_update_idp_immutable_attributes(self):
    """Update IdP's immutable parameters.

        Expect HTTP BAD REQUEST.

        """
    default_resp = self._create_default_idp()
    default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
    idp_id = default_idp.get('id')
    self.assertIsNotNone(idp_id)
    body = self._http_idp_input()
    body['id'] = uuid.uuid4().hex
    body['protocols'] = [uuid.uuid4().hex, uuid.uuid4().hex]
    url = self.base_url(suffix=idp_id)
    self.patch(url, body={'identity_provider': body}, expected_status=http.client.BAD_REQUEST)