from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
def test_get_access_token(self):
    """Test case when an access token can be successfully obtain."""
    client_name = 'client_name_test'
    app_cred = self._create_app_cred(self.user_id, client_name)
    resp = self._get_access_token(app_cred, b64str=None, headers=None, data=None, expected_status=client.OK)
    json_resp = jsonutils.loads(resp.body)
    self.assertIn('access_token', json_resp)
    self.assertEqual('Bearer', json_resp['token_type'])
    self.assertEqual(3600, json_resp['expires_in'])