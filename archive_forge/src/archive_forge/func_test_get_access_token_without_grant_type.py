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
def test_get_access_token_without_grant_type(self):
    """Test case when there is no grant_type."""
    client_name = 'client_name_test'
    app_cred = self._create_app_cred(self.user_id, client_name)
    data = {}
    error = 'invalid_request'
    error_description = 'The parameter grant_type is required.'
    resp = self._get_access_token(app_cred, b64str=None, headers=None, data=data, expected_status=client.BAD_REQUEST)
    json_resp = jsonutils.loads(resp.body)
    LOG.debug(f'error: {json_resp.get('error')}')
    LOG.debug(f'error_description: {json_resp.get('error_description')}')
    self.assertEqual(error, json_resp.get('error'))
    self.assertEqual(error_description, json_resp.get('error_description'))