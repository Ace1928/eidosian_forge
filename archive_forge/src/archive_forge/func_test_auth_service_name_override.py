from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_auth_service_name_override(self):
    self.set_http_response(status_code=200)
    self.service_connection.auth_service_name = 'service_override'
    self.service_connection.create_queue('my_queue')
    self.assertIn('us-east-1/service_override/aws4_request', self.actual_request.headers['Authorization'])