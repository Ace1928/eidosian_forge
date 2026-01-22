import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_message_is_required(self):
    self.set_http_response(status_code=200)
    with self.assertRaises(TypeError):
        self.service_connection.publish(topic='topic', subject='subject')