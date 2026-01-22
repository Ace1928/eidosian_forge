from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_sigv4_presign_response_headers(self):
    self.config = {'s3': {'use-sigv4': True}}
    conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host='s3.amazonaws.com')
    response_headers = {'response-content-disposition': 'attachment; filename="file.ext"'}
    url = conn.generate_url_sigv4(86400, 'GET', bucket='examplebucket', key='test.txt', response_headers=response_headers)
    self.assertIn('host', url)
    self.assertIn('response-content-disposition', url)