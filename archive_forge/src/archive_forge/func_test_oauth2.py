import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_oauth2(self):
    methods = ['oauth2_credential']
    oauth2_thumbprint = uuid.uuid4().hex
    token = fixture.V3Token(methods=methods, oauth2_thumbprint=oauth2_thumbprint)
    oauth2_credential = {'x5t#S256': oauth2_thumbprint}
    self.assertEqual(methods, token.methods)
    self.assertEqual(oauth2_credential, token.oauth2_credential)
    self.assertEqual(oauth2_thumbprint, token.oauth2_thumbprint)