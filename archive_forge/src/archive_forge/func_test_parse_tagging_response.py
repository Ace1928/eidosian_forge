from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.tagging import Tag
def test_parse_tagging_response(self):
    self.set_http_response(status_code=200)
    b = Bucket(self.service_connection, 'mybucket')
    api_response = b.get_tags()
    self.assertEqual(len(api_response), 1)
    self.assertEqual(len(api_response[0]), 2)
    self.assertEqual(api_response[0][0].key, 'Project')
    self.assertEqual(api_response[0][0].value, 'Project One')
    self.assertEqual(api_response[0][1].key, 'User')
    self.assertEqual(api_response[0][1].value, 'jsmith')