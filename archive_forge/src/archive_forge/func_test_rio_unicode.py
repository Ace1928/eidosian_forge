import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_rio_unicode(self):
    uni_data = 'オ'
    s = Stanza(foo=uni_data)
    self.assertEqual(s.get('foo'), uni_data)
    raw_lines = s.to_lines()
    self.assertEqual(raw_lines, [b'foo: ' + uni_data.encode('utf-8') + b'\n'])
    new_s = read_stanza(raw_lines)
    self.assertEqual(new_s.get('foo'), uni_data)