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
@mock.patch.object(shim_util, 'COMMANDS_SUPPORTING_ALL_HEADERS', new={'fake_shim'})
def test_translated_headers_get_added_to_final_command(self):
    with _mock_boto_config({'GSUtil': {'use_gcloud_storage': 'always', 'hidden_shim_mode': 'no_fallback'}}):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            fake_command = FakeCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['arg1', 'arg2'], headers={'Content-Type': 'fake_val'}, debug=1, trace_token=None, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
            self.assertTrue(fake_command.translate_to_gcloud_storage_if_requested())
            self.assertEqual(fake_command._translated_gcloud_storage_command, [shim_util._get_gcloud_binary_path('fake_dir'), 'objects', 'fake', 'arg1', 'arg2', '--content-type=fake_val'])