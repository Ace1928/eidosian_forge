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
def test_cat_range(self):
    """Tests cat command with various range arguments."""
    key_uri = self.CreateObject(contents=b'0123456789')
    stderr = self.RunGsUtil(['cat', '-r a-b', suri(key_uri)], return_stderr=True, expected_status=2 if self._use_gcloud_storage else 1)
    if self._use_gcloud_storage:
        self.assertIn('Expected a non-negative integer value or a range of such values instead of', stderr)
    else:
        self.assertIn('Invalid range', stderr)
    stderr = self.RunGsUtil(['cat', '-r 1-2-3', suri(key_uri)], return_stderr=True, expected_status=2 if self._use_gcloud_storage else 1)
    if self._use_gcloud_storage:
        self.assertIn('Expected a non-negative integer value or a range of such values instead of', stderr)
    else:
        self.assertIn('Invalid range', stderr)
    stderr = self.RunGsUtil(['cat', '-r 1.7-3', suri(key_uri)], return_stderr=True, expected_status=2 if self._use_gcloud_storage else 1)
    if self._use_gcloud_storage:
        self.assertIn('Expected a non-negative integer value or a range of such values instead of', stderr)
    else:
        self.assertIn('Invalid range', stderr)
    stdout = self.RunGsUtil(['cat', '-r', '-', suri(key_uri)], return_stdout=True)
    self.assertEqual('0123456789', stdout)
    stdout = self.RunGsUtil(['cat', '-r', '1000-3000', suri(key_uri)], return_stdout=True)
    self.assertEqual('', stdout)
    stdout = self.RunGsUtil(['cat', '-r', '1000-', suri(key_uri)], return_stdout=True)
    self.assertEqual('', stdout)
    stdout = self.RunGsUtil(['cat', '-r', '1-3', suri(key_uri)], return_stdout=True)
    self.assertEqual('123', stdout)
    stdout = self.RunGsUtil(['cat', '-r', '8-', suri(key_uri)], return_stdout=True)
    self.assertEqual('89', stdout)
    stdout = self.RunGsUtil(['cat', '-r', '0-0', suri(key_uri)], return_stdout=True)
    self.assertEqual('0', stdout)
    stdout = self.RunGsUtil(['cat', '-r', '-3', suri(key_uri)], return_stdout=True)
    self.assertEqual('789', stdout)