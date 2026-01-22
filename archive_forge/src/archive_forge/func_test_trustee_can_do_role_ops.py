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
def test_trustee_can_do_role_ops(self):
    resp = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
    trust = self.assertValidTrustResponse(resp)
    trust_token = self._get_trust_token(trust)
    resp = self.get('/OS-TRUST/trusts/%(trust_id)s/roles' % {'trust_id': trust['id']}, token=trust_token)
    self.assertValidRoleListResponse(resp, self.role)
    self.head('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']}, token=trust_token, expected_status=http.client.OK)
    resp = self.get('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']}, token=trust_token)
    self.assertValidRoleResponse(resp, self.role)