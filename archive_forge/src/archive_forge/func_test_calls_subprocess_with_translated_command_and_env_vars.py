from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
@mock.patch.object(os.environ, 'copy', return_value={'old_key': 'old_value'})
@mock.patch.object(subprocess, 'run', autospec=True)
def test_calls_subprocess_with_translated_command_and_env_vars(self, mock_run, mock_environ_copy):
    command_instance = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['-z', 'opt1', '-r', 'arg1', 'arg2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
    with util.SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'no_fallback')]):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            command_instance._translated_env_variables = {'new_key': 'new_value'}
            command_instance._translated_gcloud_storage_command = ['gcloud', 'foo']
            actual_return_code = command_instance.run_gcloud_storage()
            mock_run.assert_called_once_with(['gcloud', 'foo'], env={'old_key': 'old_value', 'new_key': 'new_value'})
            mock_environ_copy.assert_called_once_with()
            self.assertEqual(actual_return_code, mock_run.return_value.returncode)