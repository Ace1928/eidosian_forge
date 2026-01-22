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
def test_user_name_and_id_in_federation_token(self):
    r = self._issue_unscoped_token(assertion='EMPLOYEE_ASSERTION')
    self.assertEqual(mapping_fixtures.EMPLOYEE_ASSERTION['UserName'], r.user['name'])
    self.assertNotEqual(r.user['name'], r.user_id)
    r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_EMPLOYEE)
    token = r.json_body['token']
    self.assertEqual(mapping_fixtures.EMPLOYEE_ASSERTION['UserName'], token['user']['name'])
    self.assertNotEqual(token['user']['name'], token['user']['id'])