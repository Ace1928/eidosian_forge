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
def test_list_projects_for_inherited_project_assignment(self):
    subproject_inherited = unit.new_project_ref(domain_id=self.domainD['id'], parent_id=self.project_inherited['id'])
    PROVIDERS.resource_api.create_project(subproject_inherited['id'], subproject_inherited)
    PROVIDERS.assignment_api.create_grant(role_id=self.role_employee['id'], group_id=self.group_employees['id'], project_id=self.project_inherited['id'], inherited_to_projects=True)
    expected_project_ids = [self.project_all['id'], self.proj_employees['id'], subproject_inherited['id']]
    for url in ('/OS-FEDERATION/projects', '/auth/projects'):
        r = self.get(url, token=self.tokens['EMPLOYEE_ASSERTION'])
        project_ids = [project['id'] for project in r.result['projects']]
        self.assertEqual(len(expected_project_ids), len(project_ids))
        for expected_project_id in expected_project_ids:
            self.assertIn(expected_project_id, project_ids, 'Projects match failed for url %s' % url)