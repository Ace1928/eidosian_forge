from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib.commands import setmeta
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
@SkipForS3('XML header keys are case-insensitive')
def test_uppercase_header(self):
    """Tests setting custom metadata with an uppercase value."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('XML header keys are case-insensitive.')
    objuri = self.CreateObject(contents=b'foo')
    self.RunGsUtil(['setmeta', '-h', 'x-%s-meta-CaSe:SeNsItIvE' % self.provider_custom_meta, suri(objuri)])
    stdout = self.RunGsUtil(['stat', suri(objuri)], return_stdout=True)
    self.assertRegex(stdout, 'CaSe:\\s+SeNsItIvE')