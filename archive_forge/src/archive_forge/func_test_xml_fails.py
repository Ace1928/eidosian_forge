from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
@SkipForJSON('Testing XML only behavior')
def test_xml_fails(self):
    boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]
    with SetBotoConfigForTest(boto_config_hmac_auth_only):
        bucket_uri = 'gs://any-bucket-name'
        stderr = self.RunGsUtil(self._set_pap_cmd + ['inherited', bucket_uri], return_stderr=True, expected_status=1)
        self.assertIn('command can only be with the Cloud Storage JSON API', stderr)
        stderr = self.RunGsUtil(self._get_pap_cmd + [bucket_uri], return_stderr=True, expected_status=1)
        self.assertIn('command can only be with the Cloud Storage JSON API', stderr)