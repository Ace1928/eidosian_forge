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
def test_website_configuration(self):
    response = self.bucket.configure_website('index.html')
    self.assertTrue(response)
    config = self.bucket.get_website_configuration()
    self.assertEqual(config, {'WebsiteConfiguration': {'IndexDocument': {'Suffix': 'index.html'}}})
    config2, xml = self.bucket.get_website_configuration_with_xml()
    self.assertEqual(config, config2)
    self.assertTrue('<Suffix>index.html</Suffix>' in xml, xml)