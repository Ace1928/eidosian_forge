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
def test_translate_headers_for_data_transfer_command_with_additional_header(self):
    """Should log a warning."""
    self._fake_command.headers = {'additional': 'header'}
    with mock.patch.object(self._fake_command.logger, 'warn', autospec=True) as mock_warning:
        self.assertEqual(self._fake_command._translate_headers(), ['--additional-headers=additional=header'])
        mock_warning.assert_called_once_with('Header additional:header cannot be translated to a gcloud storage equivalent flag. It is being treated as an arbitrary request header.')