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
@mock.patch.object(RuleProcessor, 'process')
def test_get_access_token_process_other_exception(self, mock_process):
    self._create_mapping()
    err_msg = 'Boom!'
    mock_process.side_effect = Exception(err_msg)
    cert_content = self._get_cert_content(self.client_cert)
    resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.INTERNAL_SERVER_ERROR)
    LOG.debug(resp)
    json_resp = jsonutils.loads(resp.body)
    self.assertEqual('other_error', json_resp['error'])
    self.assertEqual(err_msg, json_resp['error_description'])
    self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)