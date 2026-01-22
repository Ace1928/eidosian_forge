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
def test_persists_custom_endpoint_through_xml_sliced_download(self):
    gs_host = config.get('Credentials', 'gs_host', DEFAULT_HOST)
    if gs_host == DEFAULT_HOST:
        return
    temporary_directory = self.CreateTempDir()
    with SetBotoConfigForTest([('GSUtil', 'sliced_object_download_threshold', '1B'), ('GSUtil', 'sliced_object_download_component_size', '1B')]):
        bucket_uri = self.CreateBucket()
        key_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')
        stdout, stderr = self.RunGsUtil(['-D', 'cp', ObjectToURI(key_uri), temporary_directory], return_stdout=True, return_stderr=True)
    output = stdout + stderr
    self.assertIn(gs_host, output)
    self.assertNotIn('hostname=' + DEFAULT_HOST, output)