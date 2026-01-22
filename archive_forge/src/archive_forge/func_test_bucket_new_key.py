from mock import patch
import xml.dom.minidom
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.exception import BotoClientError
from boto.s3.connection import Location, S3Connection
from boto.s3.bucket import Bucket
from boto.s3.deletemarker import DeleteMarker
from boto.s3.key import Key
from boto.s3.multipart import MultiPartUpload
from boto.s3.prefix import Prefix
def test_bucket_new_key(self):
    self.set_http_response(status_code=200)
    bucket = self.service_connection.create_bucket('mybucket')
    key = bucket.new_key('mykey')
    self.assertEqual(key.bucket, bucket)
    self.assertEqual(key.key, 'mykey')