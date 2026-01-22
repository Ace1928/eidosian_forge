import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_describe_stacks(self):
    self.set_http_response(status_code=200)
    stacks = self.service_connection.describe_stacks('MyStack')
    self.assertEqual(len(stacks), 1)
    stack = stacks[0]
    self.assertEqual(stack.creation_time, datetime(2012, 5, 16, 22, 55, 31))
    self.assertEqual(stack.description, 'My Description')
    self.assertEqual(stack.disable_rollback, False)
    self.assertEqual(stack.stack_id, 'arn:aws:cfn:us-east-1:1:stack')
    self.assertEqual(stack.stack_status, 'CREATE_COMPLETE')
    self.assertEqual(stack.stack_name, 'MyStack')
    self.assertEqual(stack.stack_name_reason, 'REASON')
    self.assertEqual(stack.stack_status_reason, 'REASON')
    self.assertEqual(stack.timeout_in_minutes, None)
    self.assertEqual(len(stack.outputs), 1)
    self.assertEqual(stack.outputs[0].description, 'Server URL')
    self.assertEqual(stack.outputs[0].key, 'ServerURL')
    self.assertEqual(stack.outputs[0].value, 'http://url/')
    self.assertEqual(len(stack.parameters), 1)
    self.assertEqual(stack.parameters[0].key, 'MyKey')
    self.assertEqual(stack.parameters[0].value, 'MyValue')
    self.assertEqual(len(stack.capabilities), 1)
    self.assertEqual(stack.capabilities[0].value, 'CAPABILITY_IAM')
    self.assertEqual(len(stack.notification_arns), 1)
    self.assertEqual(stack.notification_arns[0].value, 'arn:aws:sns:region-name:account-name:topic-name')
    self.assertEqual(len(stack.tags), 1)
    self.assertEqual(stack.tags['MyTagKey'], 'MyTagValue')
    self.assert_request_parameters({'Action': 'DescribeStacks', 'StackName': 'MyStack', 'Version': '2010-05-15'})