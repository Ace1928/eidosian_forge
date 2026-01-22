import uuid
from keystoneauth1.extras import kerberos
from keystoneauth1 import fixture as ks_fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.kerberos import base
def test_unscoped_mapped_auth(self):
    token_id, _ = self.kerberos_mock.mock_auth_success(url=self.token_url, method='GET')
    plugin = kerberos.MappedKerberos(auth_url=self.TEST_V3_URL, protocol=self.protocol, identity_provider=self.identity_provider)
    sess = session.Session()
    tok = plugin.get_token(sess)
    self.assertEqual(token_id, tok)