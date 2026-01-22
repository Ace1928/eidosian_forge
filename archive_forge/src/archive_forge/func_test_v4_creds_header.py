import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_v4_creds_header(self):
    auth_str = 'AWS4-HMAC-SHA256 blah'
    credentials = {'host': '127.0.0.1', 'verb': 'GET', 'path': '/v1/', 'params': {}, 'headers': {'Authorization': auth_str}}
    self.assertTrue(self.signer._v4_creds(credentials))