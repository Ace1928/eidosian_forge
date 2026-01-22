from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_get_queue_with_owner_account_id_returns_queue(self):
    self.set_http_response(status_code=200)
    self.service_connection.create_queue('my_queue')
    self.service_connection.get_queue('my_queue', '599169622985')
    assert 'QueueOwnerAWSAccountId' in self.actual_request.params.keys()
    self.assertEquals(self.actual_request.params['QueueOwnerAWSAccountId'], '599169622985')