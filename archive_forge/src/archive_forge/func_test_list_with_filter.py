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
def test_list_with_filter(self):
    self.role_data_fixtures()
    events = self.get('/OS-REVOKE/events').json_body['events']
    self.assertEqual(0, len(events))
    scoped_token = self.get_scoped_token()
    headers = {'X-Subject-Token': scoped_token}
    auth = self.build_authentication_request(token=scoped_token)
    headers2 = {'X-Subject-Token': self.get_requested_token(auth)}
    self.delete('/auth/tokens', headers=headers)
    self.delete('/auth/tokens', headers=headers2)
    events = self.get('/OS-REVOKE/events').json_body['events']
    self.assertEqual(2, len(events))
    future = utils.isotime(timeutils.utcnow() + datetime.timedelta(seconds=1000))
    events = self.get('/OS-REVOKE/events?since=%s' % future).json_body['events']
    self.assertEqual(0, len(events))