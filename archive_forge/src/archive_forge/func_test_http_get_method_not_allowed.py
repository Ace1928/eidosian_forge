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
def test_http_get_method_not_allowed(self):
    resp = self.get('/s3tokens', expected_status=http.client.METHOD_NOT_ALLOWED, convert=False)
    self.assertEqual(http.client.METHOD_NOT_ALLOWED, resp.status_code)