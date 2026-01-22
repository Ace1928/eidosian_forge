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
@mock.patch.object(subprocess, 'Popen')
def test_converts_and_logs_provisioning_os_error(self, mock_Popen):
    mock_Popen.side_effect = OSError('foobar')
    with SetBotoConfigForTest([('Credentials', 'use_client_certificate', 'True'), ('Credentials', 'cert_provider_command', 'some/path')]):
        context_config.create_context_config(self.mock_logger)
        self.mock_logger.error.assert_called_once_with('Failed to provision client certificate: foobar')