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
def testRecursiveRemoveObjsInBucket(self):
    """Tests removing all objects in bucket via rm -R gs://bucket."""
    bucket_uris = [self.CreateBucket(test_objects=['f0', 'dir/f1', 'dir/nested/f2']), self.CreateBucket(test_objects=['f0', 'dir/f1', 'dir/nested/f2'])]
    for i, final_src_char in enumerate(('', '/')):
        self.RunCommand('rm', ['-R', suri(bucket_uris[i]) + final_src_char])
        try:
            self.RunCommand('ls', [suri(bucket_uris[i])])
            self.assertTrue(False)
        except NotFoundException as e:
            self.assertEqual(e.status, 404)