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
def test_raises_error_if_sub_command_mapping_is_missing(self):
    fake_with_subcommand = FakeCommandWithSubCommandWithGcloudStorageMap(command_runner=mock.ANY, args=['set', '-y', 'opt1', '-a', 'arg1', 'arg2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
    fake_with_subcommand.gcloud_storage_map = shim_util.GcloudStorageMap(gcloud_command={}, flag_map={})
    with self.assertRaisesRegex(exception.GcloudStorageTranslationError, 'Command "fake_with_sub" cannot be translated to gcloud storage because the translation mapping is missing.'):
        fake_with_subcommand.get_gcloud_storage_args()