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
@unittest.skipUnless(HAS_OPENSSL, 'signurl requires pyopenssl.')
@mock.patch.object(ServiceAccountCredentials, '__init__', side_effect=ValueError(ERROR_MESSAGE))
def testOauth2ServiceAccountFailure(self, _):
    contents = pkgutil.get_data('gslib', 'tests/test_data/test.p12')
    tmpfile = self.CreateTempFile(contents=contents)
    with SetBotoConfigForTest(getBotoCredentialsConfig(service_account_creds={'keyfile': tmpfile, 'client_id': '?'})):
        with self.assertLogs() as logger:
            with self.assertRaises(Exception) as exc:
                gcs_json_api.GcsJsonApi(None, logging.getLogger(), None, None)
            self.assertIn(ERROR_MESSAGE, str(exc.exception))
            self.assertIn(CredTypes.OAUTH2_SERVICE_ACCOUNT, logger.output[0])