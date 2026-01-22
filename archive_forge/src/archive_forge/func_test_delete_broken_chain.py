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
def test_delete_broken_chain(self):
    self.assert_user_authenticate(self.user_list[0])
    self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': self.trust_chain[0]['id']})
    for i in range(len(self.user_list) - 1):
        auth_data = self.build_authentication_request(user_id=self.user_list[i]['id'], password=self.user_list[i]['password'])
        auth_token = self.get_requested_token(auth_data)
        self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': self.trust_chain[i + 1]['id']}, token=auth_token, expected_status=http.client.NOT_FOUND)