from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import platform
import re
import six
import gslib
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.utils.unit_util import ONE_KIB
def test_minus_D_multipart_upload(self):
    """Tests that debug option does not output upload media body."""
    for file_contents in (b'a1b2c3d4', b'a1b2c3d4\n'):
        fpath = self.CreateTempFile(contents=file_contents)
        bucket_uri = self.CreateBucket()
        with SetBotoConfigForTest([('GSUtil', 'resumable_threshold', str(ONE_KIB))]):
            stderr = self.RunGsUtil(['-D', 'cp', fpath, suri(bucket_uri)], return_stderr=True)
            print('command line:' + ' '.join(['-D', 'cp', fpath, suri(bucket_uri)]))
            if self.test_api == ApiSelector.JSON:
                self.assertIn('media body', stderr)
            self.assertNotIn('a1b2c3d4', stderr)
            self.assertIn('Comparing local vs cloud md5-checksum for', stderr)
            self.assertIn('total_bytes_transferred: %d' % len(file_contents), stderr)