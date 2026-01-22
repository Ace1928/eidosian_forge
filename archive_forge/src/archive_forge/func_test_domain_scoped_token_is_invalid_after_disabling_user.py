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
def test_domain_scoped_token_is_invalid_after_disabling_user(self):
    PROVIDERS.assignment_api.create_grant(self.role['id'], user_id=self.user['id'], domain_id=self.domain['id'])
    domain_scoped_token = self._get_domain_scoped_token()
    r = self._validate_token(domain_scoped_token)
    self.assertValidDomainScopedTokenResponse(r)
    self._set_user_enabled(self.user, enabled=False)
    self._validate_token(domain_scoped_token, expected_status=http.client.NOT_FOUND)