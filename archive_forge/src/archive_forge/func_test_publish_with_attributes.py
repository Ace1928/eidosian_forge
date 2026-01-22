import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_publish_with_attributes(self):
    self.set_http_response(status_code=200)
    self.service_connection.publish(message=json.dumps({'default': 'Ignored.', 'GCM': {'data': 'goes here'}}, sort_keys=True), message_structure='json', subject='subject', target_arn='target_arn', message_attributes={'name1': {'data_type': 'Number', 'string_value': '42'}, 'name2': {'data_type': 'String', 'string_value': 'Bob'}})
    self.assert_request_parameters({'Action': 'Publish', 'TargetArn': 'target_arn', 'Subject': 'subject', 'Message': '{"GCM": {"data": "goes here"}, "default": "Ignored."}', 'MessageStructure': 'json', 'MessageAttributes.entry.1.Name': 'name1', 'MessageAttributes.entry.1.Value.DataType': 'Number', 'MessageAttributes.entry.1.Value.StringValue': '42', 'MessageAttributes.entry.2.Name': 'name2', 'MessageAttributes.entry.2.Value.DataType': 'String', 'MessageAttributes.entry.2.Value.StringValue': 'Bob'}, ignore_params_values=['Version', 'ContentType'])