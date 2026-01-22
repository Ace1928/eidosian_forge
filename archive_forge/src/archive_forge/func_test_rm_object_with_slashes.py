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
def test_rm_object_with_slashes(self):
    """Tests removing a bucket that has objects with slashes."""
    bucket_uri = self.CreateVersionedBucket()
    ouri1 = self.CreateObject(bucket_uri=bucket_uri, object_name='h/e/l//lo', contents=b'z')
    ouri2 = self.CreateObject(bucket_uri=bucket_uri, object_name='dirnoslash/foo', contents=b'z')
    ouri3 = self.CreateObject(bucket_uri=bucket_uri, object_name='dirnoslash/foo2', contents=b'z')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 3, versioned=True)
    self._RunRemoveCommandAndCheck(['rm', '-r', suri(bucket_uri)], objects_to_remove=['%s#%s' % (suri(ouri1), urigen(ouri1)), '%s#%s' % (suri(ouri2), urigen(ouri2)), '%s#%s' % (suri(ouri3), urigen(ouri3))], buckets_to_remove=[suri(bucket_uri)])