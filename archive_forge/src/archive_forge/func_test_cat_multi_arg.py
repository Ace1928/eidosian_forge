from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.cs_api_map import ApiSelector
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import unittest
from gslib.utils import cat_helper
from gslib.utils import shim_util
from unittest import mock
def test_cat_multi_arg(self):
    """Tests cat command with multiple arguments."""
    bucket_uri = self.CreateBucket()
    data1 = b'0123456789'
    data2 = b'abcdefghij'
    obj_uri1 = self.CreateObject(bucket_uri=bucket_uri, contents=data1)
    obj_uri2 = self.CreateObject(bucket_uri=bucket_uri, contents=data2)
    stdout, stderr = self.RunGsUtil(['cat', suri(obj_uri1), suri(bucket_uri) + 'nonexistent'], return_stdout=True, return_stderr=True, expected_status=1)
    self.assertIn(data1.decode('ascii'), stdout)
    if self._use_gcloud_storage:
        self.assertIn('The following URLs matched no objects or files', stderr)
    else:
        self.assertIn('NotFoundException', stderr)
    stdout, stderr = self.RunGsUtil(['cat', suri(bucket_uri) + 'nonexistent', suri(obj_uri1)], return_stdout=True, return_stderr=True, expected_status=1)
    decoded_data1 = data1.decode('ascii')
    if self._use_gcloud_storage:
        self.assertIn(decoded_data1, stdout)
        self.assertIn('The following URLs matched no objects or files', stderr)
    else:
        self.assertNotIn(decoded_data1, stdout)
        self.assertIn('NotFoundException', stderr)
    stdout = self.RunGsUtil(['cat', suri(obj_uri1), suri(obj_uri2)], return_stdout=True)
    self.assertIn(decoded_data1 + data2.decode('ascii'), stdout)