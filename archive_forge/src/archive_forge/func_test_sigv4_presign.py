from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_sigv4_presign(self):
    self.config = {'s3': {'use-sigv4': True}}
    conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host='s3.amazonaws.com')
    url = conn.generate_url_sigv4(86400, 'GET', bucket='examplebucket', key='test.txt', iso_date='20140625T000000Z')
    self.assertIn('a937f5fbc125d98ac8f04c49e0204ea1526a7b8ca058000a54c192457be05b7d', url)