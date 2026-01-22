from keystoneauth1.loading._plugins import noauth as loader
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_basic_case(self):
    self.requests_mock.get(self.TEST_URL, text='body')
    a = noauth.NoAuth()
    s = session.Session(auth=a)
    data = s.get(self.TEST_URL, authenticated=True)
    self.assertEqual(data.text, 'body')
    self.assertRequestHeaderEqual('X-Auth-Token', self.NOAUTH_TOKEN)
    self.assertIsNone(a.get_endpoint(s))