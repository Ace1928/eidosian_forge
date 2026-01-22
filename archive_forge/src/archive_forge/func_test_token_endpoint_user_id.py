from testtools import matchers
from keystoneauth1.loading._plugins import admin_token as loader
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_token_endpoint_user_id(self):
    a = token_endpoint.Token(self.TEST_URL, self.TEST_TOKEN)
    s = session.Session()
    self.assertIsNone(a.get_user_id(s))
    self.assertIsNone(a.get_project_id(s))