from tests.unit import unittest
from mock import Mock
from boto.sqs.queue import Queue
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_queue_arn(self):
    connection = Mock()
    connection.region.name = 'us-east-1'
    q = Queue(connection=connection, url='https://sqs.us-east-1.amazonaws.com/id/queuename')
    self.assertEqual(q.arn, 'arn:aws:sqs:us-east-1:id:queuename')