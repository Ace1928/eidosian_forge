from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import unittest
from gslib.utils import execution_util
from gslib.utils import stet_util
from unittest import mock
@mock.patch.object(shutil, 'move')
@mock.patch.object(execution_util, 'ExecuteExternalCommand')
def test_stet_download_runs_binary_and_replaces_temp_file(self, mock_execute_external_command, mock_move):
    fake_config_path = self.CreateTempFile()
    mock_execute_external_command.return_value = ('stdout', 'stderr')
    mock_logger = mock.Mock()
    source_url = storage_url.StorageUrlFromString('gs://bucket/obj')
    destination_url = storage_url.StorageUrlFromString('out')
    temporary_file_name = 'out_.gstmp'
    with util.SetBotoConfigForTest([('GSUtil', 'stet_binary_path', 'fake_binary_path'), ('GSUtil', 'stet_config_path', fake_config_path)]):
        stet_util.decrypt_download(source_url, destination_url, temporary_file_name, mock_logger)
    mock_execute_external_command.assert_called_once_with(['fake_binary_path', 'decrypt', '--config-file={}'.format(fake_config_path), '--blob-id=gs://bucket/obj', 'out_.gstmp', 'out_.stet_tmp'])
    mock_logger.debug.assert_called_once_with('stderr')
    mock_move.assert_called_once_with('out_.stet_tmp', 'out_.gstmp')