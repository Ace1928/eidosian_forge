from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.commands import hash
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def testHashCloudObject(self):
    """Test hash command on a cloud object."""
    obj1 = self.CreateObject(object_name='obj1', contents=_TEST_FILE_CONTENTS)
    stdout = self.RunGsUtil(['hash', '-h', suri(obj1)], return_stdout=True)
    self.assertIn('Hashes [hex]', stdout)
    if self.default_provider == 'gs':
        self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_FILE_HEX_CRC.lower(), stdout)
    self.assertIn('\tHash (md5):\t\t%s' % _TEST_FILE_HEX_MD5, stdout)
    stdout = self.RunGsUtil(['hash', suri(obj1)], return_stdout=True)
    self.assertIn('Hashes [base64]', stdout)
    if self.default_provider == 'gs':
        self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_FILE_B64_CRC, stdout)
    self.assertIn('\tHash (md5):\t\t%s' % _TEST_FILE_B64_MD5, stdout)