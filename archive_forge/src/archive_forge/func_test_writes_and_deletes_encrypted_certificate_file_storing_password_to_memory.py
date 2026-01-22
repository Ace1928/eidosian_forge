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
@mock.patch(OPEN_TO_PATCH, new_callable=mock.mock_open)
@mock.patch.object(os, 'remove')
@mock.patch.object(subprocess, 'Popen')
def test_writes_and_deletes_encrypted_certificate_file_storing_password_to_memory(self, mock_Popen, mock_remove, mock_open):
    mock_command_process = mock.Mock()
    mock_command_process.returncode = 0
    mock_command_process.communicate.return_value = (FULL_ENCRYPTED_CERT.encode(), None)
    mock_Popen.return_value = mock_command_process
    with SetBotoConfigForTest([('Credentials', 'use_client_certificate', 'True'), ('Credentials', 'cert_provider_command', 'path --print_certificate')]):
        test_config = context_config.create_context_config(mock.Mock())
        mock_open.assert_has_calls([mock.call(test_config.client_cert_path, 'w+'), mock.call().write(CERT_SECTION), mock.call().write(ENCRYPTED_KEY_SECTION)], any_order=True)
        self.assertEqual(context_config._singleton_config.client_cert_password, PASSWORD)
        context_config._singleton_config._unprovision_client_cert()
        mock_remove.assert_called_once_with(test_config.client_cert_path)