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
def test_user_gets_only_assigned_roles(self):
    response = self._issue_unscoped_token()
    self.assertValidMappedUser(render_token.render_token_response_from_model(response)['token'])
    staging_project = PROVIDERS.resource_api.get_project_by_name('Staging', self.idp['domain_id'])
    admin = unit.new_user_ref(CONF.identity.default_domain_id)
    PROVIDERS.identity_api.create_user(admin)
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], user_id=admin['id'], project_id=staging_project['id'])
    response = self._issue_unscoped_token()
    self.assertValidMappedUser(render_token.render_token_response_from_model(response)['token'])
    unscoped_token = response.id
    scope = self._scope_request(unscoped_token, 'project', staging_project['id'])
    response = self.v3_create_token(scope)
    roles = response.json_body['token']['roles']
    role_ids = [r['id'] for r in roles]
    self.assertNotIn(self.role_admin['id'], role_ids)