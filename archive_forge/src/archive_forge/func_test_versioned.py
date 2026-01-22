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
@SkipForS3('S3 lists versions in reverse order.')
def test_versioned(self):
    """Tests listing all versions with the -a flag."""
    bucket_uri = self.CreateVersionedBucket()
    object_uri1 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'foo')
    object_uri2 = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'foo2', gs_idempotent_generation=urigen(object_uri1))

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stdout = self.RunGsUtil(['du', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(stdout, '%-11s  %s\n' % (4, suri(object_uri2)))
    _Check1()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        stdout = self.RunGsUtil(['du', '-a', suri(bucket_uri)], return_stdout=True)
        self.assertSetEqual(set(stdout.splitlines()), set(['%-11s  %s#%s' % (3, suri(object_uri1), object_uri1.generation), '%-11s  %s#%s' % (4, suri(object_uri2), object_uri2.generation)]))
    _Check2()