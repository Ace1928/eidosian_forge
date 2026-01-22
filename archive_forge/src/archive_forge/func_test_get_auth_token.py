import mistralclient.auth.keystone
from mistralclient.tests.unit.v2 import base
def test_get_auth_token(self):
    auth = self.keystone._get_auth(auth_token='token', auth_url='url', project_id='project_id')
    self.assertEqual('url', auth.auth_url)
    elements = auth.get_cache_id_elements()
    self.assertIsNotNone(elements['token'])
    self.assertIsNotNone(elements['project_id'])