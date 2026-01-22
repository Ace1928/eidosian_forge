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
def test_stet_upload_uses_no_config_if_not_provided(self, mock_execute_external_command):
    mock_execute_external_command.return_value = ('stdout', 'stderr')
    mock_logger = mock.Mock()
    source_url = storage_url.StorageUrlFromString('in')
    destination_url = storage_url.StorageUrlFromString('gs://bucket/obj')
    with util.SetBotoConfigForTest([('GSUtil', 'stet_binary_path', 'fake_binary_path'), ('GSUtil', 'stet_config_path', None)]):
        with mock.patch.object(os.path, 'exists', new=mock.Mock(return_value=True)):
            out_file_url = stet_util.encrypt_upload(source_url, destination_url, mock_logger)
    self.assertEqual(out_file_url, storage_url.StorageUrlFromString('in_.stet_tmp'))
    mock_execute_external_command.assert_called_once_with(['fake_binary_path', 'encrypt', '--blob-id=gs://bucket/obj', 'in', 'in_.stet_tmp'])
    mock_logger.debug.assert_called_once_with('stderr')