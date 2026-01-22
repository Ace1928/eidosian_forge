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
def test_trust_with_implied_roles(self):
    role1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role1['id'], role1)
    role2 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role2['id'], role2)
    PROVIDERS.role_api.create_implied_role(role1['id'], role2['id'])
    PROVIDERS.assignment_api.create_grant(role_id=role1['id'], user_id=self.user_id, project_id=self.project_id)
    ref = self.redelegated_trust_ref
    ref['roles'] = [{'id': role1['id']}, {'id': role2['id']}]
    resp = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(resp)
    role_ids = [r['id'] for r in ref['roles']]
    trust_role_ids = [r['id'] for r in trust['roles']]
    self.assertEqual(role_ids, trust_role_ids)
    auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust['id'])
    resp = self.post('/auth/tokens', body=auth_data)
    trust_token_role_ids = [r['id'] for r in resp.json['token']['roles']]
    self.assertEqual(sorted(role_ids), sorted(trust_token_role_ids))