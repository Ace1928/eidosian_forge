import gyp.easy_xml as easy_xml
import unittest
from io import StringIO
def test_EasyXml_escaping(self):
    original = '<test>\'"\r&\nfoo'
    converted = "&lt;test&gt;'&quot;&#xD;&amp;&#xA;foo"
    converted_apos = converted.replace("'", '&apos;')
    self.assertEqual(easy_xml.XmlToString(['test3', {'a': original}, original]), '<?xml version="1.0" encoding="utf-8"?><test3 a="%s">%s</test3>' % (converted, converted_apos))