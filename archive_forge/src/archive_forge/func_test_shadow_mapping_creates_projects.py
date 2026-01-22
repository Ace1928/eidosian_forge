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
def test_shadow_mapping_creates_projects(self):
    projects = PROVIDERS.resource_api.list_projects()
    for project in projects:
        self.assertNotIn(project['name'], self.expected_results)
    response = self._issue_unscoped_token()
    self.assertValidMappedUser(render_token.render_token_response_from_model(response)['token'])
    unscoped_token = response.id
    response = self.get('/auth/projects', token=unscoped_token)
    projects = response.json_body['projects']
    for project in projects:
        project = PROVIDERS.resource_api.get_project_by_name(project['name'], self.idp['domain_id'])
        self.assertIn(project['name'], self.expected_results)