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
def testHashHexFormat(self):
    tmp_file = self.CreateTempFile(contents=_TEST_FILE_CONTENTS)
    stdout = self.RunCommand('hash', args=['-h', tmp_file], return_stdout=True)
    self.assertIn('Hashes [hex]', stdout)
    self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_FILE_HEX_CRC, stdout)
    self.assertIn('\tHash (md5):\t\t%s' % _TEST_FILE_HEX_MD5, stdout)