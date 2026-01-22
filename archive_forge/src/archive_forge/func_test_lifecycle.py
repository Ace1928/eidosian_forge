from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
def test_lifecycle(self):
    lifecycle = Lifecycle()
    lifecycle.add_rule('myid', '', 'Enabled', 30)
    self.assertTrue(self.bucket.configure_lifecycle(lifecycle))
    response = self.bucket.get_lifecycle_config()
    self.assertEqual(len(response), 1)
    actual_lifecycle = response[0]
    self.assertEqual(actual_lifecycle.id, 'myid')
    self.assertEqual(actual_lifecycle.prefix, '')
    self.assertEqual(actual_lifecycle.status, 'Enabled')
    self.assertEqual(actual_lifecycle.transition, [])