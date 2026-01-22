import unittest
import os
from boto.exception import BotoServerError
from boto.sts.connection import STSConnection
from boto.sts.credentials import Credentials
from boto.s3.connection import S3Connection
def test_decode_authorization_message(self):
    c = STSConnection()
    try:
        creds = c.decode_authorization_message('b94d27b9934')
    except BotoServerError as err:
        self.assertEqual(err.status, 400)
        self.assertIn('InvalidAuthorizationMessageException', err.body)