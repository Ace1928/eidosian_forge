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
def test_roles_in_idp_domain_can_be_assigned_from_mapping(self):
    PROVIDERS.role_api.delete_role(self.member_role['id'])
    member_role_ref = unit.new_role_ref(name='member', domain_id=self.idp['domain_id'])
    PROVIDERS.role_api.create_role(member_role_ref['id'], member_role_ref)
    response = self._issue_unscoped_token()
    user_id = response.user_id
    unscoped_token = response.id
    response = self.get('/auth/projects', token=unscoped_token)
    projects = response.json_body['projects']
    staging_project = PROVIDERS.resource_api.get_project_by_name('Staging', self.idp['domain_id'])
    for project in projects:
        self.assertNotEqual(project['name'], 'Staging')
    domain_role_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user_id, project_id=staging_project['id'], strip_domain_roles=False)
    self.assertEqual(staging_project['id'], domain_role_assignments[0]['project_id'])
    self.assertEqual(user_id, domain_role_assignments[0]['user_id'])