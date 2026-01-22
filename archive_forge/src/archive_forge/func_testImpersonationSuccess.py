from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from gslib.cloud_api import AccessDeniedException
from gslib.cred_types import CredTypes
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
from datetime import datetime
@unittest.skipUnless(SERVICE_ACCOUNT, 'Test requires test_impersonate_service_account.')
@SkipForS3('Tests only uses gs credentials.')
@SkipForXML('Tests only run on JSON API.')
def testImpersonationSuccess(self):
    with SetBotoConfigForTest([('Credentials', 'gs_impersonate_service_account', SERVICE_ACCOUNT)]):
        stderr = self.RunGsUtil(['ls', 'gs://pub'], return_stderr=True)
        self.assertIn('API calls will be executed as [%s' % SERVICE_ACCOUNT, stderr)