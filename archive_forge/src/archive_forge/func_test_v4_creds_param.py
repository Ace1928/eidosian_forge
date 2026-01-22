import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_v4_creds_param(self):
    credentials = {'host': '127.0.0.1', 'verb': 'GET', 'path': '/v1/', 'params': {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256'}, 'headers': {}}
    self.assertTrue(self.signer._v4_creds(credentials))