from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import time
from gslib.command import CreateOrGetGsutilLogger
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import TAB_COMPLETE_CACHE_TTL
from gslib.tab_complete import TabCompletionCache
import gslib.tests.testcase as testcase
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.boto_util import GetTabCompletionCacheFilename
def test_prefix_caching_boundary(self):
    """Tests tab completion prefix caching not spanning directory boundaries.

    If the tab completion prefix is an extension of the cached prefix, but is
    not within the same bucket/sub-directory then the cached results should not
    be used.
    """
    with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
        object_uri = self.CreateObject(object_name='subdir/subobj', contents=b'test data')
        cached_prefix = '%s://%s/' % (self.default_provider, object_uri.bucket_name)
        cached_results = ['%s://%s/subdir' % (self.default_provider, object_uri.bucket_name)]
        _WriteTabCompletionCache(cached_prefix, cached_results)
        request = '%s://%s/subdir/' % (self.default_provider, object_uri.bucket_name)
        expected_result = '%s://%s/subdir/subobj' % (self.default_provider, object_uri.bucket_name)
        completer = CloudObjectCompleter(self.MakeGsUtilApi())
        results = completer(request)
        self.assertEqual([expected_result], results)