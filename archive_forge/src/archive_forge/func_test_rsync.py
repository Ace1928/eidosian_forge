from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retry_util import Retry
from gslib.utils.constants import UTF8
def test_rsync(self):
    req_pays_bucket_uri = self.CreateBucket(test_objects=2)
    self._set_requester_pays(req_pays_bucket_uri)
    bucket_uri = self.CreateBucket(test_objects=1)
    self._run_requester_pays_test(['rsync', '-d', suri(req_pays_bucket_uri), suri(self.requester_pays_bucket_uri)])
    bucket_uri1 = self.CreateBucket(test_objects=2)
    bucket_uri2 = self.CreateBucket(test_objects=1)
    self._run_non_requester_pays_test(['rsync', '-d', suri(bucket_uri1), suri(bucket_uri2)])