from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
def test_fails_for_missing_obj(self):
    bucket_uri = self.CreateVersionedBucket()
    stderr = self.RunGsUtil(['rm', '-a', '%s' % suri(bucket_uri, 'foo')], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        no_url_matched_target = no_url_matched_target = 'The following URLs matched no objects or files:\n-%s'
    else:
        no_url_matched_target = NO_URLS_MATCHED_TARGET
    self.assertIn(no_url_matched_target % suri(bucket_uri, 'foo'), stderr)