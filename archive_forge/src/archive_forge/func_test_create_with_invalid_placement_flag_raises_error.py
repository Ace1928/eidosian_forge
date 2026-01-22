from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
import boto
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retention_util import SECONDS_IN_DAY
from gslib.utils.retention_util import SECONDS_IN_MONTH
from gslib.utils.retention_util import SECONDS_IN_YEAR
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
@SkipForXML('The --placement flag only works for GCS JSON API.')
def test_create_with_invalid_placement_flag_raises_error(self):
    bucket_name = self.MakeTempName('bucket')
    bucket_uri = boto.storage_uri('gs://%s' % bucket_name.lower(), suppress_consec_slashes=False)
    stderr = self.RunGsUtil(['mb', '--placement', 'invalid_reg1,invalid_reg2', suri(bucket_uri)], return_stderr=True, expected_status=1)
    self.assertRegex(stderr, '.*400.*(Invalid custom placement config|One or more unrecognized regions in dual-region, received: INVALID_REG1, INVALID_REG2).*')