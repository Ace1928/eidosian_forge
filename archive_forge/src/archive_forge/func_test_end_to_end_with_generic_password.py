import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def test_end_to_end_with_generic_password(self):
    self.requests_mock.get(self.TEST_ROOT_URL, json=fixture.DiscoveryList(self.TEST_ROOT_URL), headers={'Content-Type': 'application/json'})
    self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, content=bytes(k2k_fixtures.ECP_ENVELOPE, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=200)
    self.requests_mock.register_uri('POST', self.SP_URL, content=bytes(k2k_fixtures.TOKEN_BASED_ECP, 'latin-1'), headers={'Content-Type': 'application/vnd.paos+xml'}, status_code=302)
    self.requests_mock.register_uri('GET', self.SP_AUTH_URL, json=k2k_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': k2k_fixtures.UNSCOPED_TOKEN_HEADER})
    self.stub_url('POST', ['auth', 'tokens'], headers={'X-Subject-Token': uuid.uuid4().hex}, json=self.token_v3)
    plugin = identity.Password(self.TEST_ROOT_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id='default')
    k2kplugin = self.get_plugin(base_plugin=plugin)
    self.assertEqual(k2k_fixtures.UNSCOPED_TOKEN_HEADER, k2kplugin.get_token(self.session))