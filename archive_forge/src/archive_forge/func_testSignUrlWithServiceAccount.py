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
@unittest.skipUnless(SERVICE_ACCOUNT, 'Test requires test_impersonate_service_account.')
@SkipForS3('Tests only uses gs credentials.')
@SkipForXML('Tests only run on JSON API.')
def testSignUrlWithServiceAccount(self):
    with SetBotoConfigForTest([('Credentials', 'gs_impersonate_service_account', SERVICE_ACCOUNT)]):
        stdout, stderr = self.RunGsUtil(['signurl', '-r', 'us-east1', '-u', 'gs://pub'], return_stdout=True, return_stderr=True)
    self.assertIn('https://storage.googleapis.com/pub', stdout)
    self.assertIn('All API calls will be executed as [%s]' % SERVICE_ACCOUNT, stderr)