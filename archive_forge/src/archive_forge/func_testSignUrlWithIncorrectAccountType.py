from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
from datetime import timedelta
import os
import pkgutil
import boto
import gslib.commands.signurl
from gslib.commands.signurl import HAVE_OPENSSL
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.iamcredentials_api import IamcredentailsApi
from gslib.impersonation_credentials import ImpersonationCredentials
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import (SkipForS3, SkipForXML)
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
import gslib.tests.signurl_signatures as sigs
from oauth2client import client
from oauth2client.service_account import ServiceAccountCredentials
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('Tests only uses gs credentials.')
@SkipForXML('Tests only run on JSON API.')
def testSignUrlWithIncorrectAccountType(self):
    """Tests the _GenSignedUrl with incorrect account type.

    Test that GenSignedUrl function with 'use_service_account' set to True
    and a service account not used for credentials raises an error.
    """
    expected = sigs.TEST_SIGN_URL_PUT_WITH_SERVICE_ACCOUNT
    duration = timedelta(seconds=3600)
    mock_api_delegator = self._get_mock_api_delegator()
    json_api = mock_api_delegator._GetApi('gs')
    mock_credentials = mock.Mock(spec=client.OAuth2Credentials)
    mock_credentials.service_account_email = 'fake_service_account_email'
    json_api.credentials = mock_credentials
    with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
        self.assertRaises(CommandException, gslib.commands.signurl._GenSignedUrl, None, api=mock_api_delegator, use_service_account=True, provider='gs', client_id=self.client_email, method='PUT', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='us-east1', content_type='')