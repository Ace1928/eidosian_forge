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
def test_folder_objects_deleted_with_double_wildcard(self):
    """Test for 'rm -r' of a folder with a dir_$folder$ marker."""
    bucket_uri = self.CreateVersionedBucket()
    key_uri = self.StorageUriCloneReplaceName(bucket_uri, 'abc/o1')
    self.StorageUriSetContentsFromString(key_uri, 'foobar')
    folder_uri = self.StorageUriCloneReplaceName(bucket_uri, 'abc_$folder$')
    self.StorageUriSetContentsFromString(folder_uri, '')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 2, versioned=True)
    self._RunRemoveCommandAndCheck(['rm', '-r', '%s' % suri(bucket_uri, '**')], objects_to_remove=['%s#%s' % (suri(key_uri), urigen(key_uri)), '%s#%s' % (suri(folder_uri), urigen(folder_uri))])
    self.AssertNObjectsInBucket(bucket_uri, 0, versioned=True)
    bucket_uri.get_location(validate=False)