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
def test_create_unlimited_use_trust(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, remaining_uses=None, role_ids=[self.role_id])
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(r, ref)
    r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
    auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'])
    r = self.v3_create_token(auth_data)
    token = r.headers.get('X-Subject-Token')
    auth_data = self.build_authentication_request(token=token, trust_id=trust['id'])
    r = self.v3_create_token(auth_data)
    r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
    trust = r.result.get('trust')
    self.assertIsNone(trust['remaining_uses'])