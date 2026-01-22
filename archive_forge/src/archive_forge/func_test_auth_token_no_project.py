from cinderclient.contrib import noauth
from cinderclient.tests.unit import utils
def test_auth_token_no_project(self):
    auth_token = 'user:user'
    plugin = noauth.CinderNoAuthPlugin('user')
    self.assertEqual(auth_token, plugin.auth_token)