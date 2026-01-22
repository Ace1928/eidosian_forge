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
def test_get_not_found(self):
    stderr = self.RunGsUtil(['hmac', 'get', 'GOOG1234DNE'], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('HTTPError 404: Access ID not found in project', stderr)
    else:
        self.assertIn('404 Access ID not found', stderr)