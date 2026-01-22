from tests.unit import unittest
from boto.sqs.message import MHMessage
from boto.sqs.message import RawMessage
from boto.sqs.message import Message
from boto.sqs.bigmessage import BigMessage
from boto.exception import SQSDecodeError
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_encode_bytes_message(self):
    message = Message()
    body = b'\x00\x01\x02\x03\x04\x05'
    message.set_body(body)
    self.assertEqual(message.get_body_encoded(), 'AAECAwQF')