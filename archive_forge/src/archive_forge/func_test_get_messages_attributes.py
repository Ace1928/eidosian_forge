import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def test_get_messages_attributes(self):
    conn = SQSConnection()
    current_timestamp = int(time.time())
    test = self.create_temp_queue(conn)
    time.sleep(65)
    self.put_queue_message(test)
    self.assertEqual(test.count(), 1)
    msgs = test.get_messages(num_messages=1, attributes='All')
    for msg in msgs:
        self.assertEqual(msg.attributes['ApproximateReceiveCount'], '1')
        first_rec = msg.attributes['ApproximateFirstReceiveTimestamp']
        first_rec = int(first_rec) / 1000
        self.assertTrue(first_rec >= current_timestamp)
    self.put_queue_message(test)
    self.assertEqual(test.count(), 1)
    msgs = test.get_messages(num_messages=1, attributes='ApproximateReceiveCount')
    for msg in msgs:
        self.assertEqual(msg.attributes['ApproximateReceiveCount'], '1')
        with self.assertRaises(KeyError):
            msg.attributes['ApproximateFirstReceiveTimestamp']