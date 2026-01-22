import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_generate_v2_SHA1(self):
    """Test generate function for v2 signature, SHA1."""
    credentials = {'host': '127.0.0.1', 'verb': 'GET', 'path': '/v1/', 'params': {'SignatureVersion': '2', 'AWSAccessKeyId': self.access}}
    self.signer.hmac_256 = None
    signature = self.signer.generate(credentials)
    expected = 'ZqCxMI4ZtTXWI175743mJ0hy/Gc='
    self.assertEqual(signature, expected)