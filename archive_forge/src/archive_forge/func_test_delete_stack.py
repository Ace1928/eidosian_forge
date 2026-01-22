import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_delete_stack(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_stack('stack_name')
    self.assertEqual(api_response, json.loads(self.default_body().decode('utf-8')))
    self.assert_request_parameters({'Action': 'DeleteStack', 'ContentType': 'JSON', 'StackName': 'stack_name', 'Version': '2010-05-15'})