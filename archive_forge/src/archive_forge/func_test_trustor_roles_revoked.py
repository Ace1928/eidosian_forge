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
def test_trustor_roles_revoked(self):
    self.assert_user_authenticate(self.user_list[0])
    PROVIDERS.assignment_api.remove_role_from_user_and_project(self.user_id, self.project_id, self.role_id)
    for i in range(len(self.user_list[1:])):
        trustee = self.user_list[i]
        auth_data = self.build_authentication_request(user_id=trustee['id'], password=trustee['password'])
        token = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(token=token, trust_id=self.trust_chain[i - 1]['id'])
        self.v3_create_token(auth_data, expected_status=http.client.FORBIDDEN)