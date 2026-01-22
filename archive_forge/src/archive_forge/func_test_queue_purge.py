import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def test_queue_purge(self):
    conn = SQSConnection()
    test = self.create_temp_queue(conn)
    time.sleep(65)
    for x in range(0, 4):
        self.put_queue_message(test)
    self.assertEqual(test.count(), 4)
    conn.purge_queue(test)
    self.assertEqual(test.count(), 0)