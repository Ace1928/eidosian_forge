import http.client as http_client
import json
import logging
import ssl
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import testresources
from keystoneauth1 import access
from keystoneauth1 import exceptions as ksa_exceptions
from keystonemiddleware import oauth2_mtls_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit.test_oauth2_token_middleware \
from keystonemiddleware.tests.unit.test_oauth2_token_middleware \
from keystonemiddleware.tests.unit import utils
@mock.patch.object(access, 'create')
def test_keystonemiddleware_exceptiton(self, mock_create):
    except_msg = 'Unrecognized auth response'
    mock_create.side_effect = Exception(except_msg)
    token = self.examples.v3_OAUTH2_CREDENTIAL
    resp = self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages')
    self.assertUnauthorizedResp(resp)
    self.assertIn('Invalid token contents.', self.logger.output)
    self.assertIn('Invalid OAuth2.0 certificate-bound access token: %s' % 'Token authorization failed', self.logger.output)