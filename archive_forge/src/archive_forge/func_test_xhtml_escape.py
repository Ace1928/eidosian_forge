import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_xhtml_escape(self):
    tests = [('<foo>', '&lt;foo&gt;'), ('<foo>', '&lt;foo&gt;'), (b'<foo>', b'&lt;foo&gt;'), ('<>&"\'', '&lt;&gt;&amp;&quot;&#x27;'), ('&amp;', '&amp;amp;'), ('<é>', '&lt;é&gt;'), (b'<\xc3\xa9>', b'&lt;\xc3\xa9&gt;')]
    for unescaped, escaped in tests:
        self.assertEqual(utf8(xhtml_escape(unescaped)), utf8(escaped))
        self.assertEqual(utf8(unescaped), utf8(xhtml_unescape(escaped)))