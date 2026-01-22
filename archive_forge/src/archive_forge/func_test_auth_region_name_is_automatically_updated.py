from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_auth_region_name_is_automatically_updated(self):
    region = SQSRegionInfo(name='us-west-2', endpoint='us-west-2.queue.amazonaws.com')
    self.service_connection = SQSConnection(https_connection_factory=self.https_connection_factory, aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', region=region)
    self.initialize_service_connection()
    self.set_http_response(status_code=200)
    self.service_connection.create_queue('my_queue')
    self.assertIn('us-west-2/sqs/aws4_request', self.actual_request.headers['Authorization'])