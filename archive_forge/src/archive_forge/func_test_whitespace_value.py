import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_whitespace_value(self):
    s = Stanza(space=' ', tabs='\t\t\t', combo='\n\t\t\n')
    self.assertEqual(s.to_string(), b'combo: \n\t\t\t\n\t\nspace:  \ntabs: \t\t\t\n')
    s2 = read_stanza(s.to_lines())
    self.assertEqual(s, s2)
    self.rio_file_stanzas([s])