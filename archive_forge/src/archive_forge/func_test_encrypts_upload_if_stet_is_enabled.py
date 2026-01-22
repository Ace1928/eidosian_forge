from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import stat
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils import temporary_file_util
def test_encrypts_upload_if_stet_is_enabled(self):
    object_uri = self.CreateObject()
    test_file = self.CreateTempFile(contents='will be rewritten')
    stderr = self.RunGsUtil(['-o', 'GSUtil:stet_binary_path={}'.format(self.stet_binary_path), '-o', 'GSUtil:stet_config_path={}'.format(self.stet_config_path), 'cp', '--stet', test_file, suri(object_uri)], return_stderr=True)
    self.assertNotIn('/4.0 B]', stderr)
    stdout = self.RunGsUtil(['cat', suri(object_uri)], return_stdout=True)
    self.assertIn('subcommand: encrypt', stdout)
    self.assertIn('config file: --config-file={}'.format(self.stet_config_path), stdout)
    self.assertIn('blob id: --blob-id={}'.format(suri(object_uri)), stdout)
    self.assertIn('in file: {}'.format(test_file), stdout)
    self.assertIn('out file: {}_.stet_tmp'.format(test_file), stdout)
    self.assertFalse(os.path.exists(temporary_file_util.GetStetTempFileName(storage_url.StorageUrlFromString(test_file))))