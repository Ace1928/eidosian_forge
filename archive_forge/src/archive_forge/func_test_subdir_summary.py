from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
def test_subdir_summary(self):
    """Tests summary listing with the -s flag on a subdirectory."""
    bucket_uri1, _ = self._create_nested_subdir()
    bucket_uri2, _ = self._create_nested_subdir()
    subdir1 = suri(bucket_uri1, 'sub1材')
    subdir2 = suri(bucket_uri2, 'sub1材')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        stdout = self.RunGsUtil(['du', '-s', subdir1, subdir2], return_stdout=True)
        self.assertSetEqual(set(stdout.splitlines()), set(['%-11s  %s' % (18, subdir1), '%-11s  %s' % (18, subdir2)]))
    _Check()