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
def test_create_project_token_with_disabled_project_domain_fails(self):
    domain = unit.new_domain_ref()
    domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], self.role_id)
    domain['enabled'] = False
    PROVIDERS.resource_api.update_domain(domain['id'], domain)
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=project['id'])
    self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_id=domain['id'])
    self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)