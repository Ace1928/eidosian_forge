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
@mock.patch.object(Manager, 'issue_token')
def test_get_access_token_issue_token_ks_error_401(self, mock_issue_token):
    self._create_mapping()
    err_msg = 'Boom!'
    mock_issue_token.side_effect = exception.Unauthorized(err_msg)
    cert_content = self._get_cert_content(self.client_cert)
    resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
    LOG.debug(resp)
    json_resp = jsonutils.loads(resp.body)
    self.assertEqual('invalid_client', json_resp['error'])
    self.assertEqual('The request you have made requires authentication.', json_resp['error_description'])