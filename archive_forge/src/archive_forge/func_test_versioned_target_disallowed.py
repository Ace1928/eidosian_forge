from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from six.moves import xrange
from six.moves import range
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import unittest
from gslib.project_id import PopulateProjectId
def test_versioned_target_disallowed(self):
    stderr = self.RunGsUtil(['compose', 'gs://b/o1', 'gs://b/o2', 'gs://b/o3#1234'], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Verison-specific URLs are not valid destinations because composing always results in creating an object with the latest generation.', stderr)
    else:
        self.assertIn('CommandException: A version-specific URL (%s) cannot be the destination for gsutil compose - abort.' % 'gs://b/o3#1234', stderr)