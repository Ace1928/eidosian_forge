from tests.compat import unittest
from boto.s3.connection import S3Connection
from boto.s3 import connect_to_region
def testSuccessWithHostOverrideRegion(self):
    connect_args = dict({'host': 's3.amazonaws.com'})
    connection = connect_to_region('us-west-2', **connect_args)
    self.assertEquals('s3.amazonaws.com', connection.host)
    self.assertIsInstance(connection, S3Connection)