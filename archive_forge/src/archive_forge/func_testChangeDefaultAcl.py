from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import six
from gslib.commands import defacl
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as case
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def testChangeDefaultAcl(self):
    """Tests defacl ch."""
    bucket = self.CreateBucket()
    test_regex = self._MakeScopeRegex('OWNER', 'group', self.GROUP_TEST_ADDRESS)
    test_regex2 = self._MakeScopeRegex('READER', 'group', self.GROUP_TEST_ADDRESS)
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertNotRegex(json_text, test_regex)
    self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':FC', suri(bucket)])
    json_text2 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertRegex(json_text2, test_regex)
    self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':READ', suri(bucket)])
    json_text3 = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertRegex(json_text3, test_regex2)
    stderr = self.RunGsUtil(self._defacl_ch_prefix + ['-g', self.GROUP_TEST_ADDRESS + ':WRITE', suri(bucket)], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('WRITER is not a valid value', stderr)
    else:
        self.assertIn('WRITER cannot be set as a default object ACL', stderr)