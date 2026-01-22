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
def test_get_system_roles_with_domain_scoped_token(self):
    path = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': self.role_id}
    self.put(path=path)
    project_scoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_id)
    r = self.post('/auth/tokens', body=project_scoped_request)
    project_scoped_token = r.headers.get('X-Subject-Token')
    self.assertValidProjectScopedTokenResponse(r)
    response = self.get('/auth/system', token=project_scoped_token)
    self.assertTrue(response.json_body['system'][0]['all'])
    self.head('/auth/system', token=project_scoped_token, expected_status=http.client.OK)