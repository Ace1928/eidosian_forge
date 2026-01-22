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
@SkipForS3('No composite object or crc32c support for S3.')
def testHashCompositeObject(self):
    """Test hash command on a composite object (which only has crc32c)."""
    bucket = self.CreateBucket()
    obj1 = self.CreateObject(bucket_uri=bucket, object_name='obj1', contents=_TEST_FILE_CONTENTS)
    obj2 = self.CreateObject(bucket_uri=bucket, object_name='tmp', contents=_TEST_COMPOSITE_ADDED_CONTENTS)
    self.RunGsUtil(['compose', suri(obj1), suri(obj2), suri(obj1)])
    stdout = self.RunGsUtil(['hash', '-h', suri(obj1)], return_stdout=True)
    self.assertIn('Hashes [hex]', stdout)
    self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_COMPOSITE_HEX_CRC.lower(), stdout)
    stdout = self.RunGsUtil(['hash', suri(obj1)], return_stdout=True)
    self.assertIn('Hashes [base64]', stdout)
    self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_COMPOSITE_B64_CRC, stdout)