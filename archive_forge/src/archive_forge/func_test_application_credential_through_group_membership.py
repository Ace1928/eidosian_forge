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
def test_application_credential_through_group_membership(self):
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    group1 = unit.new_group_ref(domain_id=self.domain_id)
    group1 = PROVIDERS.identity_api.create_group(group1)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.assignment_api.create_grant(self.role_id, group_id=group1['id'], project_id=self.project_id)
    app_cred = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'secret': uuid.uuid4().hex, 'user_id': user1['id'], 'project_id': self.project_id, 'description': uuid.uuid4().hex, 'roles': [{'id': self.role_id}]}
    app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
    auth_data = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'])
    self.v3_create_token(auth_data, expected_status=http.client.CREATED)