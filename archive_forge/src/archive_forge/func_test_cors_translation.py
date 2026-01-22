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
def test_cors_translation(self):
    """Tests cors translation for various formats."""
    json_text = self.cors_doc
    entries_list = CorsTranslation.JsonCorsToMessageEntries(json_text)
    boto_cors = CorsTranslation.BotoCorsFromMessage(entries_list)
    converted_entries_list = CorsTranslation.BotoCorsToMessage(boto_cors)
    converted_json_text = CorsTranslation.MessageEntriesToJson(converted_entries_list)
    self.assertEqual(json.loads(json_text), json.loads(converted_json_text))