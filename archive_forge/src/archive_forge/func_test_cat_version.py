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
def test_cat_version(self):
    """Tests cat command on versioned objects."""
    bucket_uri = self.CreateVersionedBucket()
    uri1 = self.CreateObject(bucket_uri=bucket_uri, contents=b'data1', gs_idempotent_generation=0)
    uri2 = self.CreateObject(bucket_uri=bucket_uri, object_name=uri1.object_name, contents=b'data2', gs_idempotent_generation=urigen(uri1))
    stdout = self.RunGsUtil(['cat', suri(uri1)], return_stdout=True)
    self.assertEqual('data2', stdout)
    stdout = self.RunGsUtil(['cat', uri1.version_specific_uri], return_stdout=True)
    self.assertEqual('data1', stdout)
    stdout = self.RunGsUtil(['cat', uri2.version_specific_uri], return_stdout=True)
    self.assertEqual('data2', stdout)
    if RUN_S3_TESTS:
        stderr = self.RunGsUtil(['cat', uri2.version_specific_uri + '23456'], return_stderr=True, expected_status=1)
        self.assertIn('BadRequestException: 400', stderr)
    else:
        stderr = self.RunGsUtil(['cat', uri2.version_specific_uri + '23'], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('The following URLs matched no objects or files:\n-{}23\n'.format(uri2.version_specific_uri), stderr)
        else:
            self.assertIn(NO_URLS_MATCHED_TARGET % uri2.version_specific_uri + '23', stderr)