import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
def test_get_template_fails(self):
    self.set_http_response(status_code=400)
    with self.assertRaises(self.service_connection.ResponseError):
        api_response = self.service_connection.get_template('stack_name')