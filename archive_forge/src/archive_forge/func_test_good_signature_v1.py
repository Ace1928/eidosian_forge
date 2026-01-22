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
def test_good_signature_v1(self):
    creds_ref = {'secret': u'b121dd41cdcc42fe9f70e572e84295aa'}
    credentials = {'token': 'UFVUCjFCMk0yWThBc2dUcGdBbVk3UGhDZmc9PQphcHBsaWNhdGlvbi9vY3RldC1zdHJlYW0KVHVlLCAxMSBEZWMgMjAxMiAyMTo0MTo0MSBHTVQKL2NvbnRfczMvdXBsb2FkZWRfZnJvbV9zMy50eHQ=', 'signature': 'IL4QLcLVaYgylF9iHj6Wb8BGZsw='}
    self.assertIsNone(s3tokens.S3Resource._check_signature(creds_ref, credentials))