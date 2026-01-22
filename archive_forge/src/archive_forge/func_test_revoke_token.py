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
def test_revoke_token(self):
    scoped_token = self.get_scoped_token()
    headers = {'X-Subject-Token': scoped_token}
    response = self.get('/auth/tokens', headers=headers).json_body['token']
    self.delete('/auth/tokens', headers=headers)
    self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
    events_response = self.get('/OS-REVOKE/events').json_body
    self.assertValidRevokedTokenResponse(events_response, audit_id=response['audit_ids'][0])