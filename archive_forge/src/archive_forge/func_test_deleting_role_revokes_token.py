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
def test_deleting_role_revokes_token(self):
    """Test deleting a role revokes token.

        Add some additional test data, namely:

        - A third project (project C)
        - Three additional users - user4 owned by domainB and user5 and 6 owned
          by domainA (different domain ownership should not affect the test
          results, just provided to broaden test coverage)
        - User5 is a member of group1
        - Group1 gets an additional assignment - role1 on projectB as well as
          its existing role1 on projectA
        - User4 has role2 on Project C
        - User6 has role1 on projectA and domainA
        - This allows us to create 5 tokens by virtue of different types of
          role assignment:
          - user1, scoped to ProjectA by virtue of user role1 assignment
          - user5, scoped to ProjectB by virtue of group role1 assignment
          - user4, scoped to ProjectC by virtue of user role2 assignment
          - user6, scoped to ProjectA by virtue of user role1 assignment
          - user6, scoped to DomainA by virtue of user role1 assignment
        - role1 is then deleted
        - Check the tokens on Project A and B, and DomainA are revoked, but not
          the one for Project C

        """
    self.role_data_fixtures()
    auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
    tokenA = self.get_requested_token(auth_data)
    auth_data = self.build_authentication_request(user_id=self.user5['id'], password=self.user5['password'], project_id=self.projectB['id'])
    tokenB = self.get_requested_token(auth_data)
    auth_data = self.build_authentication_request(user_id=self.user4['id'], password=self.user4['password'], project_id=self.projectC['id'])
    tokenC = self.get_requested_token(auth_data)
    auth_data = self.build_authentication_request(user_id=self.user6['id'], password=self.user6['password'], project_id=self.projectA['id'])
    tokenD = self.get_requested_token(auth_data)
    auth_data = self.build_authentication_request(user_id=self.user6['id'], password=self.user6['password'], domain_id=self.domainA['id'])
    tokenE = self.get_requested_token(auth_data)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenA}, expected_status=http.client.OK)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenB}, expected_status=http.client.OK)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenC}, expected_status=http.client.OK)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenD}, expected_status=http.client.OK)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenE}, expected_status=http.client.OK)
    role_url = '/roles/%s' % self.role1['id']
    self.delete(role_url)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenA}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenB}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenD}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenE}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': tokenC}, expected_status=http.client.OK)