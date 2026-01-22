import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_auth_token_cross_domain_group_and_project(self):
    """Verify getting a token in cross domain group/project roles."""
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    user_foo = unit.create_user(PROVIDERS.identity_api, domain_id=test_v3.DEFAULT_DOMAIN_ID)
    role_member = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_member['id'], role_member)
    role_admin = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_admin['id'], role_admin)
    role_foo_domain1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_foo_domain1['id'], role_foo_domain1)
    role_group_domain1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_group_domain1['id'], role_group_domain1)
    new_group = unit.new_group_ref(domain_id=domain1['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    PROVIDERS.identity_api.add_user_to_group(user_foo['id'], new_group['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user_foo['id'], project_id=project1['id'], role_id=role_member['id'])
    PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=project1['id'], role_id=role_admin['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user_foo['id'], domain_id=domain1['id'], role_id=role_foo_domain1['id'])
    PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], domain_id=domain1['id'], role_id=role_group_domain1['id'])
    auth_data = self.build_authentication_request(username=user_foo['name'], user_domain_id=test_v3.DEFAULT_DOMAIN_ID, password=user_foo['password'], project_name=project1['name'], project_domain_id=domain1['id'])
    r = self.v3_create_token(auth_data)
    scoped_token = self.assertValidScopedTokenResponse(r)
    project = scoped_token['project']
    roles_ids = []
    for ref in scoped_token['roles']:
        roles_ids.append(ref['id'])
    self.assertEqual(project1['id'], project['id'])
    self.assertIn(role_member['id'], roles_ids)
    self.assertIn(role_admin['id'], roles_ids)
    self.assertNotIn(role_foo_domain1['id'], roles_ids)
    self.assertNotIn(role_group_domain1['id'], roles_ids)