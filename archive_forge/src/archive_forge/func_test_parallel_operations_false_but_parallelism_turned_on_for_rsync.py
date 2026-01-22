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
def test_parallel_operations_false_but_parallelism_turned_on_for_rsync(self):
    command = rsync.RsyncCommand(command_runner=mock.ANY, args=['arg1', 'arg2'], headers={}, debug=0, trace_token=None, parallel_operations=False, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
    with util.SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'no_fallback')]):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            command.translate_to_gcloud_storage_if_requested()
            self.assertNotIn('CLOUDSDK_STORAGE_PROCESS_COUNT', command._translated_env_variables)
            self.assertNotIn('CLOUDSDK_STORAGE_THREAD_COUNT', command._translated_env_variables)