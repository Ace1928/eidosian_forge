from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_malformed_commands(self):
    params = [('hmac create', 'requires a service account', 'argument SERVICE_ACCOUNT: Must be specified'), ('hmac create -p proj', 'requires a service account', 'argument SERVICE_ACCOUNT: Must be specified'), ('hmac delete', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac delete -p proj', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac get', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac get -p proj', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac list account1', 'unexpected arguments', 'unrecognized arguments'), ('hmac update keyname', 'state flag must be supplied', 'Exactly one of (--activate | --deactivate) must be specified.'), ('hmac update -s INACTIVE', 'requires an Access ID', 'argument ACCESS_ID: Must be specified'), ('hmac update -s INACTIVE -p proj', 'requires an Access ID', 'argument ACCESS_ID: Must be specified')]
    for command, gsutil_error_substr, gcloud_error_substr in params:
        expected_status = 2 if self._use_gcloud_storage else 1
        stderr = self.RunGsUtil(command.split(), return_stderr=True, expected_status=expected_status)
        if self._use_gcloud_storage:
            self.assertIn(gcloud_error_substr, stderr)
        else:
            self.assertIn(gsutil_error_substr, stderr)