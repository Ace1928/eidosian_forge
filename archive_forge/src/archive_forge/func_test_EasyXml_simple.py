import gyp.easy_xml as easy_xml
import unittest
from io import StringIO
def test_EasyXml_simple(self):
    self.assertEqual(easy_xml.XmlToString(['test']), '<?xml version="1.0" encoding="utf-8"?><test/>')
    self.assertEqual(easy_xml.XmlToString(['test'], encoding='Windows-1252'), '<?xml version="1.0" encoding="Windows-1252"?><test/>')