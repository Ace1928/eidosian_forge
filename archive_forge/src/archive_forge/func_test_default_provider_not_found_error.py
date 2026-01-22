from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import subprocess
from unittest import mock
import six
from gslib import context_config
from gslib import exception
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import SetBotoConfigForTest
@mock.patch('os.path.exists', new=mock.Mock(return_value=False))
def test_default_provider_not_found_error(self):
    with SetBotoConfigForTest([('Credentials', 'use_client_certificate', 'True'), ('Credentials', 'cert_provider_command', None), ('GSUtil', 'state_dir', self.CreateTempDir())]):
        context_config.create_context_config(self.mock_logger)
        self.mock_logger.error.assert_called_once_with('Failed to provision client certificate: Client certificate provider file not found.')