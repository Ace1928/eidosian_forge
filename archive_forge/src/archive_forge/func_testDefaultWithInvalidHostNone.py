from tests.compat import unittest
from boto.s3.connection import S3Connection
from boto.s3 import connect_to_region
def testDefaultWithInvalidHostNone(self):
    connect_args = dict({'host': None})
    connection = connect_to_region('us-east-1', **connect_args)
    self.assertEquals('s3.amazonaws.com', connection.host)
    self.assertIsInstance(connection, S3Connection)