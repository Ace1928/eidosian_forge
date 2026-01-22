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
def test_revoke_token_from_token(self):
    unscoped_token = self.get_requested_token(self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password']))
    project_scoped_token = self.get_requested_token(self.build_authentication_request(token=unscoped_token, project_id=self.projectA['id']))
    domain_scoped_token = self.get_requested_token(self.build_authentication_request(token=unscoped_token, domain_id=self.domainA['id']))
    self.delete('/auth/tokens', headers={'X-Subject-Token': project_scoped_token})
    self.head('/auth/tokens', headers={'X-Subject-Token': project_scoped_token}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': unscoped_token}, expected_status=http.client.OK)
    self.head('/auth/tokens', headers={'X-Subject-Token': domain_scoped_token}, expected_status=http.client.OK)
    self.delete('/auth/tokens', headers={'X-Subject-Token': domain_scoped_token})
    self.head('/auth/tokens', headers={'X-Subject-Token': domain_scoped_token}, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers={'X-Subject-Token': unscoped_token}, expected_status=http.client.OK)