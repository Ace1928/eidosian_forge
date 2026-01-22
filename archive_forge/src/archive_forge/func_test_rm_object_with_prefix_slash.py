from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
@SkipForS3('The boto lib used for S3 does not handle objects starting with slashes if we use V4 signature')
def test_rm_object_with_prefix_slash(self):
    """Tests removing a bucket that has an object starting with slash.

    The boto lib used for S3 does not handle objects starting with slashes
    if we use V4 signature. Hence we are testing objects with prefix
    slashes separately.
    """
    bucket_uri = self.CreateVersionedBucket()
    ouri1 = self.CreateObject(bucket_uri=bucket_uri, object_name='/dirwithslash/foo', contents=b'z')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 1, versioned=True)
    self._RunRemoveCommandAndCheck(['rm', '-r', suri(bucket_uri)], objects_to_remove=['%s#%s' % (suri(ouri1), urigen(ouri1))], buckets_to_remove=[suri(bucket_uri)])