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
def test_tagging(self):
    tagging = '\n            <Tagging>\n              <TagSet>\n                 <Tag>\n                   <Key>tagkey</Key>\n                   <Value>tagvalue</Value>\n                 </Tag>\n              </TagSet>\n            </Tagging>\n        '
    self.bucket.set_xml_tags(tagging)
    response = self.bucket.get_tags()
    self.assertEqual(response[0][0].key, 'tagkey')
    self.assertEqual(response[0][0].value, 'tagvalue')
    self.bucket.delete_tags()
    try:
        self.bucket.get_tags()
    except S3ResponseError as e:
        self.assertEqual(e.code, 'NoSuchTagSet')
    except Exception as e:
        self.fail('Wrong exception raised (expected S3ResponseError): %s' % e)
    else:
        self.fail('Expected S3ResponseError, but no exception raised.')