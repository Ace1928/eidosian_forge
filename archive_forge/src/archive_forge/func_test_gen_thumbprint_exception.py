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
@mock.patch.object(ssl, 'DER_cert_to_PEM_cert')
def test_gen_thumbprint_exception(self, mock_DER_cert_to_PEM_cert):
    except_msg = 'Boom!'
    mock_DER_cert_to_PEM_cert.side_effect = Exception(except_msg)
    token = self.examples.v3_OAUTH2_CREDENTIAL
    resp = self.call_middleware(headers=get_authorization_header(token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages')
    self.assertUnauthorizedResp(resp)
    self.assertIn(except_msg, self.logger.output)