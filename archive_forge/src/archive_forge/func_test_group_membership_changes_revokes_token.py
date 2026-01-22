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
def test_group_membership_changes_revokes_token(self):
    """Test add/removal to/from group revokes token.

        Test Plan:

        - Get a token for user1, scoped to ProjectA
        - Get a token for user2, scoped to ProjectA
        - Remove user1 from group1
        - Check token for user1 is no longer valid
        - Check token for user2 is still valid, even though
          user2 is also part of group1
        - Add user2 to group2
        - Check token for user2 is now no longer valid

        """
    auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
    token1 = self.get_requested_token(auth_data)
    auth_data = self.build_authentication_request(user_id=self.user2['id'], password=self.user2['password'], project_id=self.projectA['id'])
    token2 = self.get_requested_token(auth_data)
    self.head('/auth/tokens', headers={'X-Subject-Token': token1}, expected_status=http.client.OK)
    self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)
    self.delete('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group1['id'], 'user_id': self.user1['id']})
    self.head('/auth/tokens', headers={'X-Subject-Token': token1}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)
    self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group2['id'], 'user_id': self.user2['id']})
    self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)