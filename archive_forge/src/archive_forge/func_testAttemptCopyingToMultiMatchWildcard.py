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
def testAttemptCopyingToMultiMatchWildcard(self):
    """Attempts to copy where dst wildcard matches >1 obj."""
    src_bucket_uri = self.CreateBucket(test_objects=2)
    try:
        self.RunCommand('cp', [suri(src_bucket_uri, 'obj0'), suri(src_bucket_uri, '*')])
        self.fail('Did not get expected CommandException')
    except CommandException as e:
        self.assertNotEqual(e.reason.find('must match exactly 1 URL'), -1)