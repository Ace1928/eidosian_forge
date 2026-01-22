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
def test_remove_all_versions_recursive_on_bucket(self):
    """Test that 'rm -r' works on bucket."""
    bucket_uri = self.CreateVersionedBucket()
    k1_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo')
    k2_uri = self.StorageUriCloneReplaceName(bucket_uri, 'foo2')
    self.StorageUriSetContentsFromString(k1_uri, 'bar')
    self.StorageUriSetContentsFromString(k2_uri, 'bar2')
    k1g1 = urigen(k1_uri)
    k2g1 = urigen(k2_uri)
    self.StorageUriSetContentsFromString(k1_uri, 'baz')
    self.StorageUriSetContentsFromString(k2_uri, 'baz2')
    k1g2 = urigen(k1_uri)
    k2g2 = urigen(k2_uri)
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 4, versioned=True)
    self._RunRemoveCommandAndCheck(['rm', '-r', suri(bucket_uri)], objects_to_remove=['%s#%s' % (suri(k1_uri), k1g1), '%s#%s' % (suri(k1_uri), k1g2), '%s#%s' % (suri(k2_uri), k2g1), '%s#%s' % (suri(k2_uri), k2g2)], buckets_to_remove=[suri(bucket_uri)])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        stderr = self.RunGsUtil(['ls', '-a', suri(bucket_uri)], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            if self._use_gcloud_storage:
                self.assertTrue('not found: 404' in stderr or 'NoSuchBucket' in stderr)
        else:
            self.assertIn('bucket does not exist', stderr)
    _Check()