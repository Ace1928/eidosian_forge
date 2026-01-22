from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import os
import six
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetDummyProjectForUnitTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import copy_helper
from gslib.utils import system_util
def testCopyingCompressedFileToBucket(self):
    """Tests copying one file with compression to a bucket."""
    src_file = self.CreateTempFile(contents=b'plaintext', file_name='f2.txt')
    dst_bucket_uri = self.CreateBucket()
    self.RunCommand('cp', ['-z', 'txt', src_file, suri(dst_bucket_uri)])
    actual = list(self._test_wildcard_iterator(suri(dst_bucket_uri, '*')).IterAll(expand_top_level_buckets=True))
    self.assertEqual(1, len(actual))
    actual_obj = actual[0].root_object
    self.assertEqual('f2.txt', actual_obj.name)
    self.assertEqual('gzip', actual_obj.contentEncoding)
    stdout = self.RunCommand('cat', [suri(dst_bucket_uri, 'f2.txt')], return_stdout=True)
    f = gzip.GzipFile(fileobj=six.BytesIO(six.ensure_binary(stdout)), mode='rb')
    try:
        self.assertEqual(f.read(), b'plaintext')
    finally:
        f.close()