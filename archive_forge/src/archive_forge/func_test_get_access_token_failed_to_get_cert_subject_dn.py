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
@mock.patch.object(utils, 'get_certificate_subject_dn')
def test_get_access_token_failed_to_get_cert_subject_dn(self, mock_get_certificate_subject_dn):
    self._create_mapping()
    mock_get_certificate_subject_dn.side_effect = exception.ValidationError('Boom!')
    cert_content = self._get_cert_content(self.client_cert)
    resp = self._get_access_token(client_id=self.oauth2_user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
    self.assertUnauthorizedResp(resp)
    self.assertIn('Get OAuth2.0 Access Token API: failed to get the subject DN from the certificate.', self.log_fix.output)