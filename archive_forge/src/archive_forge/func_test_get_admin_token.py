import requests
import testtools.matchers
from keystone.tests.functional import core as functests
def test_get_admin_token(self):
    token = self.get_scoped_admin_token()
    self.assertIsNotNone(token)