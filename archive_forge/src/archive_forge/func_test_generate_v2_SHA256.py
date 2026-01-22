import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_generate_v2_SHA256(self):
    """Test generate function for v2 signature, SHA256."""
    credentials = {'host': '127.0.0.1', 'verb': 'GET', 'path': '/v1/', 'params': {'SignatureVersion': '2', 'AWSAccessKeyId': self.access}}
    signature = self.signer.generate(credentials)
    expected = 'odsGmT811GffUO0Eu13Pq+xTzKNIjJ6NhgZU74tYX/w='
    self.assertEqual(signature, expected)