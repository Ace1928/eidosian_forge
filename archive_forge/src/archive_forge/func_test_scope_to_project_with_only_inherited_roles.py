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
def test_scope_to_project_with_only_inherited_roles(self):
    """Try to scope token whose only roles are inherited."""
    r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_INHERITED_FROM_CUSTOMER)
    token_resp = r.result['token']
    self._check_project_scoped_token_attributes(token_resp, self.project_inherited['id'])
    roles_ref = [self.role_customer]
    projects_ref = self.project_inherited
    self._check_projects_and_roles(token_resp, roles_ref, projects_ref)
    self.assertValidMappedUser(token_resp)