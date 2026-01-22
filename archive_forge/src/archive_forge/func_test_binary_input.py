from boto.compat import json
from boto.kms.layer1 import KMSConnection
from tests.unit import AWSMockServiceTestCase
def test_binary_input(self):
    """
        This test ensures that binary is base64 encoded when it is sent to
        the service.
        """
    self.set_http_response(status_code=200)
    data = b'\x00\x01\x02\x03\x04\x05'
    self.service_connection.encrypt(key_id='foo', plaintext=data)
    body = json.loads(self.actual_request.body.decode('utf-8'))
    self.assertEqual(body['Plaintext'], 'AAECAwQF')