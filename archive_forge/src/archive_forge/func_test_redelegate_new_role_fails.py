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
def test_redelegate_new_role_fails(self):
    r = self.post('/OS-TRUST/trusts', body={'trust': self.redelegated_trust_ref})
    trust = self.assertValidTrustResponse(r)
    trust_token = self._get_trust_token(trust)
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    PROVIDERS.assignment_api.create_grant(role_id=role['id'], user_id=self.user_id, project_id=self.project_id)
    self.chained_trust_ref['roles'] = [{'id': role['id']}]
    with mock.patch.object(policy, 'enforce', return_value=True):
        self.post('/OS-TRUST/trusts', body={'trust': self.chained_trust_ref}, token=trust_token, expected_status=http.client.FORBIDDEN)