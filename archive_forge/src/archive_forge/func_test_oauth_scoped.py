import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_oauth_scoped(self):
    access_id = uuid.uuid4().hex
    consumer_id = uuid.uuid4().hex
    token = fixture.V3Token(oauth_access_token_id=access_id, oauth_consumer_id=consumer_id)
    oauth = token['token']['OS-OAUTH1']
    self.assertEqual(access_id, token.oauth_access_token_id)
    self.assertEqual(access_id, oauth['access_token_id'])
    self.assertEqual(consumer_id, token.oauth_consumer_id)
    self.assertEqual(consumer_id, oauth['consumer_id'])