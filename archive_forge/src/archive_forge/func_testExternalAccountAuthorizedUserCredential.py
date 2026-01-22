from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import GceAssertionCredentials
from google_reauth import reauth_creds
from gslib import gcs_json_api
from gslib import gcs_json_credentials
from gslib.cred_types import CredTypes
from gslib.exception import CommandException
from gslib.tests import testcase
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.utils.wrapped_credentials import WrappedCredentials
import logging
from oauth2client.service_account import ServiceAccountCredentials
import pkgutil
from six import add_move, MovedModule
from six.moves import mock
def testExternalAccountAuthorizedUserCredential(self):
    contents = pkgutil.get_data('gslib', 'tests/test_data/test_external_account_authorized_user_credentials.json')
    tmpfile = self.CreateTempFile(contents=contents)
    with SetBotoConfigForTest(getBotoCredentialsConfig(external_account_authorized_user_creds=tmpfile)):
        client = gcs_json_api.GcsJsonApi(None, None, None, None)
        self.assertIsInstance(client.credentials, WrappedCredentials)