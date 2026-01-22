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
def test_create_idp_authorization_ttl(self):
    keys_to_check = list(self.idp_keys)
    keys_to_check.append('authorization_ttl')
    body = self.default_body.copy()
    body['description'] = uuid.uuid4().hex
    body['authorization_ttl'] = 10080
    resp = self._create_default_idp(body)
    expected = body.copy()
    self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=expected)