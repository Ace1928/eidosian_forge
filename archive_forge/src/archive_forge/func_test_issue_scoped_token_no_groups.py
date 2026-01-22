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
def test_issue_scoped_token_no_groups(self):
    """Verify that token without groups cannot get scoped to project.

        This test is required because of bug 1677723.
        """
    r = self._issue_unscoped_token(assertion='USER_NO_GROUPS_ASSERTION')
    token_groups = r.federated_groups
    self.assertEqual(0, len(token_groups))
    unscoped_token = r.id
    self.proj_employees
    admin = unit.new_user_ref(CONF.identity.default_domain_id)
    PROVIDERS.identity_api.create_user(admin)
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], user_id=admin['id'], project_id=self.proj_employees['id'])
    scope = self._scope_request(unscoped_token, 'project', self.proj_employees['id'])
    self.v3_create_token(scope, expected_status=http.client.UNAUTHORIZED)