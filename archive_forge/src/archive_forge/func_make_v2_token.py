from keystoneauth1 import fixture as ksa_fixture
from requests_mock.contrib import fixture
from openstackclient.tests.unit import test_shell
from openstackclient.tests.unit import utils
def make_v2_token(req_mock):
    """Create an Identity v2 token and register the responses"""
    token = ksa_fixture.V2Token(tenant_name=test_shell.DEFAULT_PROJECT_NAME, user_name=test_shell.DEFAULT_USERNAME)
    req_mock.register_uri('GET', V2_AUTH_URL, json=V2_VERSION_RESP, status_code=200)
    req_mock.register_uri('POST', V2_AUTH_URL + 'tokens', json=token, status_code=200)
    return token