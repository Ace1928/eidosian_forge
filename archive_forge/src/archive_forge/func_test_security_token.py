from tests.unit import unittest
from boto.sts.connection import STSConnection
from tests.unit import AWSMockServiceTestCase
def test_security_token(self):
    self.assertEqual('token', self.service_connection.provider.security_token)