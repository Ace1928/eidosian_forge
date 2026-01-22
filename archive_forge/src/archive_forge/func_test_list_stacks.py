import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_list_stacks(self):
    self.set_http_response(status_code=200)
    stacks = self.service_connection.list_stacks(['CREATE_IN_PROGRESS'], next_token='next_token')
    self.assertEqual(len(stacks), 1)
    self.assertEqual(stacks[0].stack_id, 'arn:aws:cfn:us-east-1:1:stack/Test1/aa')
    self.assertEqual(stacks[0].stack_status, 'CREATE_IN_PROGRESS')
    self.assertEqual(stacks[0].stack_name, 'vpc1')
    self.assertEqual(stacks[0].creation_time, datetime(2011, 5, 23, 15, 47, 44))
    self.assertEqual(stacks[0].deletion_time, None)
    self.assertEqual(stacks[0].template_description, 'My Description.')
    self.assert_request_parameters({'Action': 'ListStacks', 'NextToken': 'next_token', 'StackStatusFilter.member.1': 'CREATE_IN_PROGRESS', 'Version': '2010-05-15'})