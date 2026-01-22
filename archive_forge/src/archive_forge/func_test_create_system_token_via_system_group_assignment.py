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
def test_create_system_token_via_system_group_assignment(self):
    ref = {'group': unit.new_group_ref(domain_id=CONF.identity.default_domain_id)}
    group = self.post('/groups', body=ref).json_body['group']
    path = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': self.role_id}
    self.put(path=path)
    path = '/groups/%(group_id)s/users/%(user_id)s' % {'group_id': group['id'], 'user_id': self.user['id']}
    self.put(path=path)
    auth_request_body = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], system=True)
    response = self.v3_create_token(auth_request_body)
    self.assertValidSystemScopedTokenResponse(response)
    token = response.headers.get('X-Subject-Token')
    self._validate_token(token)