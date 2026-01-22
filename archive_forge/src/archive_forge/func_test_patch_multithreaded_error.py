from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
def test_patch_multithreaded_error(self):
    """See TestIamSet.test_set_multithreaded_error."""

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stderr = self.RunGsUtil(['-m', 'iam', 'ch', '-r', '%s:legacyObjectReader' % self.user, 'gs://%s' % self.nonexistent_bucket_name, self.bucket.uri], return_stderr=True, expected_status=1)
        if self._use_gcloud_storage:
            self.assertIn('not found: 404.', stderr)
        else:
            self.assertIn('BucketNotFoundException', stderr)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        object_iam_string = self.RunGsUtil(['iam', 'get', self.object.uri], return_stdout=True)
        object2_iam_string = self.RunGsUtil(['iam', 'get', self.object2.uri], return_stdout=True)
        self.assertEqualsPoliciesString(self.object_iam_string, object_iam_string)
        self.assertEqualsPoliciesString(self.object_iam_string, object2_iam_string)
    _Check1()
    _Check2()