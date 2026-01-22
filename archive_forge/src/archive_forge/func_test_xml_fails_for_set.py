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
@SkipForJSON('Testing XML only behavior.')
def test_xml_fails_for_set(self):
    boto_config_hmac_auth_only = [('Credentials', 'gs_oauth2_refresh_token', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_service_key_file_password', None), ('Credentials', 'gs_access_key_id', 'dummykey'), ('Credentials', 'gs_secret_access_key', 'dummysecret')]
    with SetBotoConfigForTest(boto_config_hmac_auth_only):
        bucket_uri = 'gs://any-bucket-name'
        stderr = self.RunGsUtil(['rpo', 'set', 'default', bucket_uri], return_stderr=True, expected_status=1)
        self.assertIn('command can only be with the Cloud Storage JSON API', stderr)