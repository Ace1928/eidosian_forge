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
@mock.patch.object(execution_util, 'ExecuteExternalCommand')
def test_stet_upload_runs_with_binary_from_path_with_correct_settings(self, mock_execute_external_command):
    fake_config_path = self.CreateTempFile()
    temporary_path_directory = self.CreateTempDir()
    fake_stet_binary_path = self.CreateTempFile(tmpdir=temporary_path_directory, file_name='stet')
    previous_path = os.getenv('PATH')
    os.environ['PATH'] += os.path.pathsep + temporary_path_directory
    mock_execute_external_command.return_value = ('stdout', 'stderr')
    mock_logger = mock.Mock()
    source_url = storage_url.StorageUrlFromString('in')
    destination_url = storage_url.StorageUrlFromString('gs://bucket/obj')
    with util.SetBotoConfigForTest([('GSUtil', 'stet_binary_path', None), ('GSUtil', 'stet_config_path', fake_config_path)]):
        out_file_url = stet_util.encrypt_upload(source_url, destination_url, mock_logger)
    self.assertEqual(out_file_url, storage_url.StorageUrlFromString('in_.stet_tmp'))
    mock_execute_external_command.assert_called_once_with([fake_stet_binary_path, 'encrypt', '--config-file={}'.format(fake_config_path), '--blob-id=gs://bucket/obj', 'in', 'in_.stet_tmp'])
    mock_logger.debug.assert_called_once_with('stderr')
    os.environ['PATH'] = previous_path