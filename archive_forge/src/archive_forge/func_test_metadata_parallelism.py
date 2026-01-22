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
def test_metadata_parallelism(self):
    """Ensure that custom metadata works in the multi-thread/process case."""
    bucket_uri = self.CreateBucket(test_objects=2)
    self.AssertNObjectsInBucket(bucket_uri, 2)
    self.RunGsUtil(['setmeta', '-h', 'x-%s-meta-abc:123' % self.provider_custom_meta, suri(bucket_uri, '**')])