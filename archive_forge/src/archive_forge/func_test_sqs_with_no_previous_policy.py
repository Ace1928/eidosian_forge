import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_sqs_with_no_previous_policy(self):
    self.set_http_response(status_code=200)
    queue = Mock()
    queue.get_attributes.return_value = {}
    queue.arn = 'arn:aws:sqs:us-east-1:idnum:queuename'
    self.service_connection.subscribe_sqs_queue('topic_arn', queue)
    self.assert_request_parameters({'Action': 'Subscribe', 'ContentType': 'JSON', 'Endpoint': 'arn:aws:sqs:us-east-1:idnum:queuename', 'Protocol': 'sqs', 'TopicArn': 'topic_arn', 'Version': '2010-03-31'}, ignore_params_values=[])
    actual_policy = json.loads(queue.set_attribute.call_args[0][1])
    self.assertEqual(len(actual_policy['Statement']), 1)