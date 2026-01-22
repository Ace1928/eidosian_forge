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
def test_set_lifecycle_and_reset(self):
    """Tests setting and turning off lifecycle configuration."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    fpath = self.CreateTempFile(tmpdir=tmpdir, contents=self.lifecycle_doc.encode('ascii'))
    self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
    stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.lifecycle_json_obj)
    fpath = self.CreateTempFile(tmpdir=tmpdir, contents=self.empty_doc1.encode('ascii'))
    self.RunGsUtil(['lifecycle', 'set', fpath, suri(bucket_uri)])
    stdout = self.RunGsUtil(['lifecycle', 'get', suri(bucket_uri)], return_stdout=True)
    if self._use_gcloud_storage:
        self.assertIn(self.empty_lifecycle_config, stdout)
    else:
        self.assertIn(self.no_lifecycle_config, stdout)