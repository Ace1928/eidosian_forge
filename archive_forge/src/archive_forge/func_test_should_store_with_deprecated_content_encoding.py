import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
def test_should_store_with_deprecated_content_encoding(self):
    """DEPRECATION WARNING

        Manually setting the payload_content_encoding is deprecated and will be
        removed in a future release.
        """
    data = {'secret_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    encoded_payload = base64.b64encode(b'F\x130\x89f\x8e\xd9\xa1\x0e\x1f\r\xf67uu\x8b').decode('UTF-8')
    payload_content_type = 'application/octet-stream'
    payload_content_encoding = 'base64'
    secret = self.manager.create()
    secret.payload = encoded_payload
    secret.payload_content_type = payload_content_type
    secret.payload_content_encoding = payload_content_encoding
    secret.store()
    secret_req = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(encoded_payload, secret_req['payload'])
    self.assertEqual(payload_content_type, secret_req['payload_content_type'])
    self.assertEqual(payload_content_encoding, secret_req['payload_content_encoding'])