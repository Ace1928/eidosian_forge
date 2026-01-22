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
def test_create_project_scoped_token_fails_if_domain_name_unsafe(self):
    """Verify authenticate to a project using unsafe domain name fails."""
    self.config_fixture.config(group='resource', domain_name_url_safe='off')
    unsafe_name = 'i am not / safe'
    domain = unit.new_domain_ref(name=unsafe_name)
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    role_member = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_member['id'], role_member)
    PROVIDERS.assignment_api.create_grant(role_member['id'], user_id=self.user['id'], project_id=project['id'])
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_name=domain['name'])
    self.v3_create_token(auth_data)
    self.config_fixture.config(group='resource', project_name_url_safe='new')
    self.v3_create_token(auth_data)
    self.config_fixture.config(group='resource', domain_name_url_safe='strict')
    self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)