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
def test_get_access_token_cert_dn_missing_domain_name(self):
    self._create_mapping()
    user, user_domain, _ = self._create_project_user()
    *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), email_address=user.get('email'), domain_component=user_domain.get('id')))
    cert_content = self._get_cert_content(client_cert)
    resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content, expected_status=http.client.UNAUTHORIZED)
    self.assertUnauthorizedResp(resp)
    self.assertIn('Get OAuth2.0 Access Token API: mapping rule process failed.', self.log_fix.output)