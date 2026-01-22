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
def test_recursive_bucket_rm_with_wildcarding(self):
    """Tests removing all objects and buckets matching a bucket wildcard."""
    buri_base = 'gsutil-test-%s' % self.GetTestMethodName()
    buri_base = buri_base[:MAX_BUCKET_LENGTH - 20]
    buri_base = '%s-%s' % (buri_base, self.MakeRandomTestString())
    buri_base = 'aaa-' + buri_base
    buri_base = util.MakeBucketNameValid(buri_base)
    buri1 = self.CreateBucket(bucket_name='%s-tbuck1' % buri_base)
    buri2 = self.CreateBucket(bucket_name='%s-tbuck2' % buri_base)
    buri3 = self.CreateBucket(bucket_name='%s-tb3' % buri_base)
    ouri1 = self.CreateObject(bucket_uri=buri1, object_name='o1', contents=b'z')
    ouri2 = self.CreateObject(bucket_uri=buri2, object_name='o2', contents=b'z')
    self.CreateObject(bucket_uri=buri3, object_name='o3', contents=b'z')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(buri1, 1)
        self.AssertNObjectsInBucket(buri2, 1)
        self.AssertNObjectsInBucket(buri3, 1)
    self._RunRemoveCommandAndCheck(['rm', '-r', '%s://%s-tbu*' % (self.default_provider, buri_base)], objects_to_remove=['%s#%s' % (suri(ouri1), urigen(ouri1)), '%s#%s' % (suri(ouri2), urigen(ouri2))], buckets_to_remove=[suri(buri1), suri(buri2)])
    self.AssertNObjectsInBucket(buri3, 1)