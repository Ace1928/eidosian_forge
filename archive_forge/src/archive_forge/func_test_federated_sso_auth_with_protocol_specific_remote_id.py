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
def test_federated_sso_auth_with_protocol_specific_remote_id(self):
    self.config_fixture.config(group=self.PROTOCOL, remote_id_attribute=self.PROTOCOL_REMOTE_ID_ATTR)
    environment = {self.PROTOCOL_REMOTE_ID_ATTR: self.REMOTE_IDS[0], 'QUERY_STRING': 'origin=%s' % self.ORIGIN}
    environment.update(mapping_fixtures.EMPLOYEE_ASSERTION)
    with self.make_request(environ=environment):
        resp = auth_api.AuthFederationWebSSOResource._perform_auth(self.PROTOCOL)
    self.assertIn(self.TRUSTED_DASHBOARD.encode('utf-8'), resp.data)