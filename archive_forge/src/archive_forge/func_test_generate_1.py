import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_generate_1(self):
    """Test generate function for v1 signature."""
    credentials = {'host': '127.0.0.1', 'verb': 'GET', 'path': '/v1/', 'params': {'SignatureVersion': '1', 'AWSAccessKeyId': self.access}}
    signature = self.signer.generate(credentials)
    expected = 'VRnoQH/EhVTTLhwRLfuL7jmFW9c='
    self.assertEqual(signature, expected)