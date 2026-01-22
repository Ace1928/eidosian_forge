from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import posixpath
from xml.dom.minidom import parseString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import CorsTranslation
def test_set_multi_non_null_cors(self):
    """Tests setting different CORS configurations."""
    bucket1_uri = self.CreateBucket()
    bucket2_uri = self.CreateBucket()
    fpath = self.CreateTempFile(contents=self.cors_doc)
    self.RunGsUtil(self._set_cmd_prefix + [fpath, suri(bucket1_uri), suri(bucket2_uri)])
    stdout = self.RunGsUtil(self._get_cmd_prefix + [suri(bucket1_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.cors_json_obj)
    stdout = self.RunGsUtil(self._get_cmd_prefix + [suri(bucket2_uri)], return_stdout=True)
    self.assertEqual(json.loads(stdout), self.cors_json_obj)