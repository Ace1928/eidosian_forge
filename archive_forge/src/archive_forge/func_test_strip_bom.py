import sys
import codecs
from unittest import TestCase
import simplejson as json
from simplejson.compat import unichr, text_type, b, BytesIO
def test_strip_bom(self):
    content = u'こんにちわ'
    json_doc = codecs.BOM_UTF8 + b(json.dumps(content))
    self.assertEqual(json.load(BytesIO(json_doc)), content)
    for doc in (json_doc, json_doc.decode('utf8')):
        self.assertEqual(json.loads(doc), content)