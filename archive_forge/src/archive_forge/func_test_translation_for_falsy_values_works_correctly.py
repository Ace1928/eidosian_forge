from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
from unittest import mock
from xml.dom.minidom import parseString
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import LifecycleTranslation
from gslib.utils import shim_util
def test_translation_for_falsy_values_works_correctly(self):
    bucket_uri = self.CreateBucket()
    fpath = self.CreateTempFile(contents=self.lifecycle_with_falsy_condition_values.encode('ascii'))
    self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
    stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
    self.assertRegex(stdout, '"age":\\s+0')
    self.assertRegex(stdout, '"isLive":\\s+false')
    self.assertRegex(stdout, '"numNewerVersions":\\s+0')