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
def test_raises_error_if_cloudsdk_root_dir_is_none(self):
    with util.SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True'), ('GSUtil', 'hidden_shim_mode', 'no_fallback')]):
        with util.SetEnvironmentForTest({'CLOUDSDK_ROOT_DIR': None}):
            with self.assertRaisesRegex(exception.CommandException, 'CommandException: Requested to use "gcloud storage" but the gcloud binary path cannot be found. This might happen if you attempt to use gsutil that was not installed via Cloud SDK. You can manually set the `CLOUDSDK_ROOT_DIR` environment variable to point to the google-cloud-sdk installation directory to resolve the issue. Alternatively, you can set `use_gcloud_storage=False` to disable running the command using gcloud storage.'):
                self._fake_command.translate_to_gcloud_storage_if_requested()