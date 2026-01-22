from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.cs_api_map import ApiSelector
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_MD5
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY2_SHA256_B64
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
@SkipForS3("'Archived time' is a GS-specific response field.")
@SkipForXML("XML API only supports 'DeletedTime' response field when making a GET Bucket request to list all objects, which is heavy overhead when the real intent is just a HEAD Object call.")
def test_versioned_stat_output(self):
    """Tests stat output of an outdated object under version control."""
    bucket_uri = self.CreateVersionedBucket()
    old_object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'z')
    self.CreateObject(bucket_uri=bucket_uri, object_name=old_object_uri.object_name, contents=b'z', gs_idempotent_generation=urigen(old_object_uri))
    stdout = self.RunGsUtil(['stat', old_object_uri.version_specific_uri], return_stdout=True)
    self.assertIn('Noncurrent time', stdout)