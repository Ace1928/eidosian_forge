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
def test_setmeta_raises_error_if_not_provided_headers(self):
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(['setmeta', suri(bucket_uri)], expected_status=1, return_stderr=True)
    self.assertIn('gsutil setmeta requires one or more headers to be provided with the -h flag. See "gsutil help setmeta" for more information.', stderr)