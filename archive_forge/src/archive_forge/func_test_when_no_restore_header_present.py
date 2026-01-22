import io
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.compat import StringIO
from boto.exception import BotoServerError
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.key import Key
def test_when_no_restore_header_present(self):
    self.set_http_response(status_code=200)
    b = Bucket(self.service_connection, 'mybucket')
    k = b.get_key('myglacierkey')
    self.assertIsNone(k.ongoing_restore)
    self.assertIsNone(k.expiry_date)