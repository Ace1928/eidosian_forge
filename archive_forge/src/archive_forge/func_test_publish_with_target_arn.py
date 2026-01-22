import json
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from mock import Mock
from boto.sns.connection import SNSConnection
def test_publish_with_target_arn(self):
    self.set_http_response(status_code=200)
    self.service_connection.publish(target_arn='target_arn', message='message', subject='subject')
    self.assert_request_parameters({'Action': 'Publish', 'TargetArn': 'target_arn', 'Subject': 'subject', 'Message': 'message'}, ignore_params_values=['Version', 'ContentType'])