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
def testAttemptCopyingMultiObjsToFile(self):
    """Attempts to copy multiple objects to a file."""
    src_bucket_uri = self.CreateBucket(test_objects=2)
    dst_file = self.CreateTempFile()
    try:
        self.RunCommand('cp', ['-R', suri(src_bucket_uri, '*'), dst_file])
        self.fail('Did not get expected CommandException')
    except CommandException as e:
        self.assertIn('must name a directory, bucket, or', e.reason)