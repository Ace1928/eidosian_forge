import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_processing_instruction_trailing_qmark(self):
    text = u'<?php echo "Foobar" ??>'
    events = list(HTMLParser(StringIO(text)))
    kind, (target, data), pos = events[0]
    self.assertEqual(Stream.PI, kind)
    self.assertEqual('php', target)
    self.assertEqual('echo "Foobar" ?', data)