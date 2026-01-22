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
def test_off_default(self):
    bucket_uri = self.CreateBucket()
    stdout = self.RunGsUtil(self._get_rp_cmd + [suri(bucket_uri)], return_stdout=True)
    self.assertEqual(stdout.strip(), '%s: Disabled' % suri(bucket_uri))