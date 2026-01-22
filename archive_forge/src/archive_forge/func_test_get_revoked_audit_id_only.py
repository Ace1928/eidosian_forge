import uuid
from keystoneauth1 import exceptions
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
def test_get_revoked_audit_id_only(self):
    sample_revoked_response = {'revoked': [{'audit_id': uuid.uuid4().hex, 'expires': '2016-01-21T15:53:52Z'}]}
    self.stub_url('GET', ['auth', 'tokens', 'OS-PKI', 'revoked'], json=sample_revoked_response)
    resp = self.client.tokens.get_revoked(audit_id_only=True)
    self.assertQueryStringIs('audit_id_only')
    self.assertEqual(sample_revoked_response, resp)