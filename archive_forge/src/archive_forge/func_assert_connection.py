import os
from tests.unit import unittest
def assert_connection(self, region, host):
    from boto.route53 import connect_to_region
    from boto.route53.connection import Route53Connection
    connection = connect_to_region(region, aws_access_key_id='foo', aws_secret_access_key='bar')
    self.assertIsInstance(connection, Route53Connection)
    self.assertEqual(connection.host, host)