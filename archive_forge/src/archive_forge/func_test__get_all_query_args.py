from mock import patch
import xml.dom.minidom
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.exception import BotoClientError
from boto.s3.connection import Location, S3Connection
from boto.s3.bucket import Bucket
from boto.s3.deletemarker import DeleteMarker
from boto.s3.key import Key
from boto.s3.multipart import MultiPartUpload
from boto.s3.prefix import Prefix
def test__get_all_query_args(self):
    bukket = Bucket()
    qa = bukket._get_all_query_args({})
    self.assertEqual(qa, '')
    qa = bukket._get_all_query_args({}, 'initial=1')
    self.assertEqual(qa, 'initial=1')
    qa = bukket._get_all_query_args({'foo': 'true'})
    self.assertEqual(qa, 'foo=true')
    qa = bukket._get_all_query_args({'foo': 'true'}, 'initial=1')
    self.assertEqual(qa, 'initial=1&foo=true')
    multiple_params = {'foo': 'true', 'bar': '☃', 'baz': u'χ', 'some_other': 'thing', 'maxkeys': 0, 'notthere': None, 'notpresenteither': ''}
    qa = bukket._get_all_query_args(multiple_params)
    self.assertEqual(qa, 'bar=%E2%98%83&baz=%CF%87&foo=true&max-keys=0&some-other=thing')
    qa = bukket._get_all_query_args(multiple_params, 'initial=1')
    self.assertEqual(qa, 'initial=1&bar=%E2%98%83&baz=%CF%87&foo=true&max-keys=0&some-other=thing')