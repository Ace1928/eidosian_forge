from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
def test_set_too_few_arguments_fails(self):
    stderr = self.RunGsUtil(self._set_pap_cmd, return_stderr=True, expected_status=1)
    self.assertIn('command requires at least', stderr)