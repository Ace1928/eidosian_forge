from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto import config
from gslib.gcs_json_api import DEFAULT_HOST
from gslib.tests import testcase
from gslib.tests.testcase import integration_testcase
from gslib.tests.util import ObjectToURI
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
@integration_testcase.SkipForJSON('XML test.')
@integration_testcase.SkipForS3('Custom endpoints not available for S3.')
def test_persists_custom_endpoint_through_resumable_upload(self):
    gs_host = config.get('Credentials', 'gs_host', DEFAULT_HOST)
    if gs_host == DEFAULT_HOST:
        return
    temporary_file = self.CreateTempFile(contents=b'foo')
    with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', '1')]):
        bucket_uri = self.CreateBucket()
        stdout, stderr = self.RunGsUtil(['-D', 'cp', temporary_file, ObjectToURI(bucket_uri)], return_stdout=True, return_stderr=True)
    output = stdout + stderr
    self.assertIn(gs_host, output)
    self.assertNotIn('hostname=' + DEFAULT_HOST, output)