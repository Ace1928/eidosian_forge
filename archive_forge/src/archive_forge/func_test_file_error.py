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
def test_file_error(self):
    key = Key()

    class CustomException(Exception):
        pass
    key.get_contents_to_file = mock.Mock(side_effect=CustomException('File blew up!'))
    with self.assertRaises(CustomException):
        key.get_contents_to_filename('foo.txt')