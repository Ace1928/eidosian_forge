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
def testDeletePermissionsWithCh(self):
    """Tests removing permissions with defacl ch."""
    bucket = self.CreateBucket()
    test_regex = self._MakeScopeRegex('OWNER', 'user', self.USER_TEST_ADDRESS)
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertNotRegex(json_text, test_regex)
    self.RunGsUtil(self._defacl_ch_prefix + ['-u', self.USER_TEST_ADDRESS + ':fc', suri(bucket)])
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertRegex(json_text, test_regex)
    self.RunGsUtil(self._defacl_ch_prefix + ['-d', self.USER_TEST_ADDRESS, suri(bucket)])
    json_text = self.RunGsUtil(self._defacl_get_prefix + [suri(bucket)], return_stdout=True)
    self.assertNotRegex(json_text, test_regex)