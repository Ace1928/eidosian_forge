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
def testMultipleConfiguredCreds(self):
    with SetBotoConfigForTest([('Credentials', 'gs_oauth2_refresh_token', 'foo'), ('Credentials', 'gs_service_client_id', 'bar'), ('Credentials', 'gs_service_key_file', 'baz'), ('Credentials', 'gs_impersonate_service_account', None)]):
        try:
            GcsJsonApi(None, self.logger, DiscardMessagesQueue())
            self.fail('Succeeded with multiple types of configured creds.')
        except CommandException as e:
            msg = str(e)
            self.assertIn('types of configured credentials', msg)
            self.assertIn(CredTypes.OAUTH2_USER_ACCOUNT, msg)
            self.assertIn(CredTypes.OAUTH2_SERVICE_ACCOUNT, msg)