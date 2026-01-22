from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import unittest
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
from gslib.pubsub_api import PubsubApi
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
def test_create_basic(self):
    """Tests the create command succeeds in normal circumstances."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('Notifications only work with the JSON API.')
    bucket_uri = self.CreateBucket()
    topic_name = self._RegisterDefaultTopicCreation(bucket_uri.bucket_name)
    stdout, stderr = self.RunGsUtil(['notification', 'create', '-f', 'json', suri(bucket_uri)], return_stdout=True, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('kind: storage#notification', stdout)
        self.assertIn(topic_name, stdout)
    else:
        self.assertIn('Created notification', stderr)
        self.assertIn(topic_name, stderr)