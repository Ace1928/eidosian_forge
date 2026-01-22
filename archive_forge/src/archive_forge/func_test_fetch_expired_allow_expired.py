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
def test_fetch_expired_allow_expired(self):
    self.config_fixture.config(group='token', expiration=10, allow_expired_window=20)
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        token = self._get_project_scoped_token()
        frozen_datetime.tick(delta=datetime.timedelta(seconds=2))
        self._validate_token(token)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=12))
        self._validate_token(token, expected_status=http.client.NOT_FOUND)
        self._validate_token(token, allow_expired=True)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=22))
        self._validate_token(token, allow_expired=True, expected_status=http.client.NOT_FOUND)