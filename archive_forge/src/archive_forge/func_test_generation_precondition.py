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
@SkipForS3('Preconditions not supported for s3 objects')
def test_generation_precondition(self):
    """Tests setting metadata with a generation precondition."""
    object_uri = self.CreateObject(contents=b'foo')
    generation = object_uri.generation
    ct = 'image/gif'
    stderr = self.RunGsUtil(['-h', 'x-goog-if-generation-match:%d' % (long(generation) + 1), 'setmeta', '-h', 'x-%s-meta-xyz:abc' % self.provider_custom_meta, '-h', 'Content-Type:%s' % ct, suri(object_uri)], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('pre-condition', stderr)
    else:
        self.assertIn('Precondition', stderr)
    self.RunGsUtil(['-h', 'x-goog-generation-match:%s' % generation, 'setmeta', '-h', 'x-%s-meta-xyz:abc' % self.provider_custom_meta, '-h', 'Content-Type:%s' % ct, suri(object_uri)])
    stdout = self.RunGsUtil(['ls', '-L', suri(object_uri)], return_stdout=True)
    self.assertRegex(stdout, 'Content-Type:\\s+%s' % ct)
    self.assertRegex(stdout, 'xyz:\\s+abc')