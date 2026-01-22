from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import unittest
from boto.storage_uri import BucketStorageUri
from gslib.cs_api_map import ApiSelector
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.gcs_json_api import GcsJsonApi
from gslib.project_id import PopulateProjectId
from gslib.tests.rewrite_helper import EnsureRewriteRestartCallbackHandler
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import unittest
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.unit_util import ONE_MIB
def test_rewrite_key_rotation_single_object(self):
    if self.test_api == ApiSelector.XML:
        return unittest.skip('Rewrite API is only supported in JSON.')
    object_uri = self.CreateObject(contents=b'bar', encryption_key=TEST_ENCRYPTION_KEY1)
    boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY2), ('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY1)]
    with SetBotoConfigForTest(boto_config_for_test):
        stderr = self.RunGsUtil(['rewrite', '-k', suri(object_uri)], return_stderr=True)
        self.assertIn(self.rotating_message, stderr)
    self.AssertObjectUsesCSEK(suri(object_uri), TEST_ENCRYPTION_KEY2)
    boto_config_for_test2 = [('GSUtil', 'decryption_key1', TEST_ENCRYPTION_KEY2)]
    with SetBotoConfigForTest(boto_config_for_test2):
        stderr = self.RunGsUtil(['rewrite', '-k', suri(object_uri)], return_stderr=True)
        self.assertIn(self.decrypting_message, stderr)
    self.AssertObjectUnencrypted(suri(object_uri))