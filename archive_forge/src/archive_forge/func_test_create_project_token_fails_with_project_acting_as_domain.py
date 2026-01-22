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
def test_create_project_token_fails_with_project_acting_as_domain(self):
    domain = unit.new_project_ref(is_domain=True)
    domain = PROVIDERS.resource_api.create_project(domain['id'], domain)
    role_member = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_member['id'], role_member)
    PROVIDERS.assignment_api.create_grant(role_member['id'], user_id=self.user['id'], domain_id=domain['id'])
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=domain['name'], project_domain_name=domain['name'])
    self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)