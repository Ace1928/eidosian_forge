from __future__ import absolute_import
import os
import textwrap
from gslib.commands.rpo import RpoCommand
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@SkipForGS('Testing S3 only behavior.')
def test_s3_fails_for_get(self):
    bucket_uri = self.CreateBucket()
    expected_status = 0 if self._use_gcloud_storage else 1
    stdout, stderr = self.RunGsUtil(['rpo', 'get', suri(bucket_uri)], return_stderr=True, return_stdout=True, expected_status=expected_status)
    if self._use_gcloud_storage:
        self.assertIn('gs://None: None', stdout)
    else:
        self.assertIn('command can only be used for GCS buckets', stderr)