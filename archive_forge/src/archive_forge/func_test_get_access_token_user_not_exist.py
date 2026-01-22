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
def test_get_access_token_user_not_exist(self):
    self._create_mapping()
    cert_content = self._get_cert_content(self.client_cert)
    user_id_not_exist = 'user_id_not_exist'
    resp = self._get_access_token(client_id=user_id_not_exist, client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
    self.assertUnauthorizedResp(resp)
    self.assertIn('Get OAuth2.0 Access Token API: the user does not exist. user id: %s' % user_id_not_exist, self.log_fix.output)