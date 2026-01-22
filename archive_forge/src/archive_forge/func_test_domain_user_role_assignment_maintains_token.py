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
def test_domain_user_role_assignment_maintains_token(self):
    """Test user-domain role assignment maintains existing token.

        Test Plan:

        - Get a token for user1, scoped to ProjectA
        - Create a grant for user1 on DomainB
        - Check token is still valid

        """
    auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
    token = self.get_requested_token(auth_data)
    self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
    grant_url = '/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s' % {'domain_id': self.domainB['id'], 'user_id': self.user1['id'], 'role_id': self.role1['id']}
    self.put(grant_url)
    self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)