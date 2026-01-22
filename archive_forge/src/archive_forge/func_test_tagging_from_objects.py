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
def test_tagging_from_objects(self):
    """Create tags from python objects rather than raw xml."""
    t = Tags()
    tag_set = TagSet()
    tag_set.add_tag('akey', 'avalue')
    tag_set.add_tag('anotherkey', 'anothervalue')
    t.add_tag_set(tag_set)
    self.bucket.set_tags(t)
    response = self.bucket.get_tags()
    tags = sorted(response[0], key=lambda tag: tag.key)
    self.assertEqual(tags[0].key, 'akey')
    self.assertEqual(tags[0].value, 'avalue')
    self.assertEqual(tags[1].key, 'anotherkey')
    self.assertEqual(tags[1].value, 'anothervalue')