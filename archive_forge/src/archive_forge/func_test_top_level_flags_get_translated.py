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
def test_top_level_flags_get_translated(self):
    """Should return True and perform the translation."""
    boto_config = {'GSUtil': {'use_gcloud_storage': 'always', 'hidden_shim_mode': 'no_fallback'}}
    with _mock_boto_config(boto_config):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['arg1', 'arg2'], headers={}, debug=3, trace_token='fake_trace_token', user_project='fake_user_project', parallel_operations=False, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
            self.assertTrue(fake_command.translate_to_gcloud_storage_if_requested())
            expected_gcloud_path = shim_util._get_gcloud_binary_path('fake_dir')
            self.assertEqual(fake_command._translated_gcloud_storage_command, [expected_gcloud_path, 'objects', 'fake', 'arg1', 'arg2', '--verbosity', 'debug', '--billing-project=fake_user_project', '--trace-token=fake_trace_token'])
            self.assertCountEqual(fake_command._translated_env_variables, {'CLOUDSDK_STORAGE_PROCESS_COUNT': '1', 'CLOUDSDK_STORAGE_THREAD_COUNT': '1', 'CLOUDSDK_METRICS_ENVIRONMENT': 'gsutil_shim', 'CLOUDSDK_STORAGE_RUN_BY_GSUTIL_SHIM': 'True'})