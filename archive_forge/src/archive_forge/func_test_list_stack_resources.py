import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_list_stack_resources(self):
    self.set_http_response(status_code=200)
    resources = self.service_connection.list_stack_resources('MyStack', next_token='next_token')
    self.assertEqual(len(resources), 2)
    self.assertEqual(resources[0].last_updated_time, datetime(2011, 6, 21, 20, 25, 57))
    self.assertEqual(resources[0].logical_resource_id, 'SampleDB')
    self.assertEqual(resources[0].physical_resource_id, 'My-db-ycx')
    self.assertEqual(resources[0].resource_status, 'CREATE_COMPLETE')
    self.assertEqual(resources[0].resource_status_reason, None)
    self.assertEqual(resources[0].resource_type, 'AWS::RDS::DBInstance')
    self.assertEqual(resources[1].last_updated_time, datetime(2011, 6, 21, 20, 29, 23))
    self.assertEqual(resources[1].logical_resource_id, 'CPUAlarmHigh')
    self.assertEqual(resources[1].physical_resource_id, 'MyStack-CPUH-PF')
    self.assertEqual(resources[1].resource_status, 'CREATE_COMPLETE')
    self.assertEqual(resources[1].resource_status_reason, None)
    self.assertEqual(resources[1].resource_type, 'AWS::CloudWatch::Alarm')
    self.assert_request_parameters({'Action': 'ListStackResources', 'NextToken': 'next_token', 'StackName': 'MyStack', 'Version': '2010-05-15'})