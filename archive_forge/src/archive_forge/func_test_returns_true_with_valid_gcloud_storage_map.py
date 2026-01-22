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
def test_returns_true_with_valid_gcloud_storage_map(self):
    """Should return True and perform the translation."""
    with _mock_boto_config({'GSUtil': {'use_gcloud_storage': 'always', 'hidden_shim_mode': 'no_fallback'}}):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            self.assertTrue(self._fake_command.translate_to_gcloud_storage_if_requested())
            expected_gcloud_path = shim_util._get_gcloud_binary_path('fake_dir')
            self.assertEqual(self._fake_command._translated_gcloud_storage_command, [expected_gcloud_path, 'objects', 'fake', '--zip', 'opt1', '-x', 'arg1', 'arg2'])