import base64
import hashlib
import hmac
import uuid
import http.client
from keystone.api import s3tokens
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_bad_response(self):
    self.post('/s3tokens', body={'credentials': {'access': self.cred_blob['access'], 'signature': base64.b64encode(b'totally not the sig').strip(), 'token': base64.b64encode(b'string to sign').strip()}}, expected_status=http.client.UNAUTHORIZED)