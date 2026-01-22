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
def test_compose_between_buckets_fails(self):
    bucket_uri_1 = self.CreateBucket()
    bucket_uri_2 = self.CreateBucket()
    object_uri1 = self.CreateObject(bucket_uri=bucket_uri_1, contents=b'1')
    object_uri2 = self.CreateObject(bucket_uri=bucket_uri_2, contents=b'2')
    composite_object_uri = self.StorageUriCloneReplaceName(bucket_uri_1, self.MakeTempName('obj'))
    stderr = self.RunGsUtil(['compose', suri(object_uri1), suri(object_uri2), suri(composite_object_uri)], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Inter-bucket composing not supported\n', stderr)
    else:
        self.assertIn('CommandException: GCS does not support inter-bucket composing.\n', stderr)